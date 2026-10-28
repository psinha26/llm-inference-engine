// CLI: ./engine --model m.bin --tokenizer tokenizer.json --prompt "..." [options]
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model.h"
#include "sampler.h"
#include "tokenizer/tokenizer.h"

namespace {

struct Args {
    std::string model, tokenizer, prompt;
    std::string tokens;       // comma-separated ids, bypasses the tokenizer (testing)
    std::string dump_logits;  // write last-prompt-position logits (float32) here
    std::string ids_out;      // write prompt + generated ids here, one per line
    int max_tokens = 256;
    int threads = 1;
    int ctx = 0;  // 0 = prompt + max_tokens
    std::string quant = "fp32";
    bool kv_cache = true;
    bool stop_at_eos = true;
    bool quiet = false;
    SamplerConfig sampler{0.0f, 0, 1.0f, 42};
};

void usage() {
    std::fprintf(stderr,
        "usage: engine --model m.bin --tokenizer tokenizer.json --prompt \"...\" [options]\n"
        "  --max-tokens N     tokens to generate (default 256)\n"
        "  --temp T           sampling temperature, 0 = greedy (default 0)\n"
        "  --top-k K          keep the K most likely tokens (default off)\n"
        "  --top-p P          nucleus sampling threshold (default off)\n"
        "  --seed S           RNG seed (default 42)\n"
        "  --threads N        worker threads (default 1)\n"
        "  --quant fp32       weight format (fp32 for now)\n"
        "  --ctx N            context length to allocate (default prompt + max-tokens)\n"
        "  --no-kv-cache      naive baseline: recompute the whole sequence every token\n"
        "  --ignore-eos       keep generating past the end-of-text token\n"
        "  --tokens 1,2,3     feed raw token ids instead of --prompt\n"
        "  --dump-logits F    write the logits after the prompt to F (float32)\n"
        "  --ids-out F        write prompt + generated token ids to F\n"
        "  --quiet            don't stream generated text\n");
    std::exit(1);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (k == "--model") a.model = val();
        else if (k == "--tokenizer") a.tokenizer = val();
        else if (k == "--prompt") a.prompt = val();
        else if (k == "--tokens") a.tokens = val();
        else if (k == "--max-tokens") a.max_tokens = std::stoi(val());
        else if (k == "--temp") a.sampler.temperature = std::stof(val());
        else if (k == "--top-k") a.sampler.top_k = std::stoi(val());
        else if (k == "--top-p") a.sampler.top_p = std::stof(val());
        else if (k == "--seed") a.sampler.seed = std::stoull(val());
        else if (k == "--threads") a.threads = std::stoi(val());
        else if (k == "--quant") a.quant = val();
        else if (k == "--ctx") a.ctx = std::stoi(val());
        else if (k == "--no-kv-cache") a.kv_cache = false;
        else if (k == "--ignore-eos") a.stop_at_eos = false;
        else if (k == "--dump-logits") a.dump_logits = val();
        else if (k == "--ids-out") a.ids_out = val();
        else if (k == "--quiet") a.quiet = true;
        else usage();
    }
    if (a.model.empty() || (a.prompt.empty() && a.tokens.empty())) usage();
    if (!a.prompt.empty() && a.tokenizer.empty()) {
        std::fprintf(stderr, "--prompt needs --tokenizer\n");
        std::exit(1);
    }
    return a;
}

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);  // bytes on macOS
#else
    return ru.ru_maxrss / 1024.0;  // kilobytes on Linux
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    try {
        if (args.quant != "fp32") throw std::runtime_error("only --quant fp32 is implemented so far");
        if (args.threads != 1) std::fprintf(stderr, "note: multithreading not implemented yet, using 1 thread\n");

        std::unique_ptr<Tokenizer> tok;
        if (!args.tokenizer.empty()) tok = std::make_unique<Tokenizer>(args.tokenizer);

        std::vector<int> seq;
        if (!args.tokens.empty()) {
            std::stringstream ss(args.tokens);
            std::string item;
            while (std::getline(ss, item, ',')) seq.push_back(std::stoi(item));
        } else {
            seq = tok->encode(args.prompt);
        }
        if (seq.empty()) throw std::runtime_error("prompt encoded to zero tokens");
        const int n_prompt = int(seq.size());

        int ctx = args.ctx > 0 ? args.ctx : n_prompt + args.max_tokens;
        Model model(args.model, ctx);
        const Config& cfg = model.config();
        if (n_prompt + args.max_tokens > model.ctx_len())
            args.max_tokens = std::max(0, model.ctx_len() - n_prompt);
        Sampler sampler(args.sampler);
        const int eos = tok ? tok->eos_id() : -1;

        if (!args.quiet && tok) {
            std::fputs(tok->decode(seq).c_str(), stdout);
            std::fflush(stdout);
        }

        // ---- prefill ----
        double t0 = now_s();
        const float* logits = nullptr;
        for (int p = 0; p < n_prompt; ++p) logits = model.forward(seq[p], p, p == n_prompt - 1);
        double t_prefill = now_s() - t0;

        if (!args.dump_logits.empty()) {
            std::ofstream f(args.dump_logits, std::ios::binary);
            f.write(reinterpret_cast<const char*>(logits), sizeof(float) * cfg.vocab_size);
        }

        // ---- decode ----
        int generated = 0, decode_steps = 0;
        double t1 = now_s();
        while (generated < args.max_tokens) {
            int next = sampler.sample(logits, cfg.vocab_size);
            seq.push_back(next);
            ++generated;
            if (!args.quiet && tok) {
                std::fputs(tok->decode_token(next).c_str(), stdout);
                std::fflush(stdout);
            }
            if (args.stop_at_eos && next == eos) break;
            if (generated == args.max_tokens) break;
            const int pos = int(seq.size()) - 1;
            if (args.kv_cache) {
                logits = model.forward(next, pos, true);
            } else {
                // Naive baseline: no reuse at all, recompute every position from scratch.
                for (int p = 0; p <= pos; ++p) logits = model.forward(seq[p], p, p == pos);
            }
            ++decode_steps;
        }
        double t_decode = now_s() - t1;
        if (!args.quiet) std::fputs("\n", stdout);

        if (!args.ids_out.empty()) {
            std::ofstream f(args.ids_out);
            for (int id : seq) f << id << "\n";
        }

        std::fprintf(stderr,
                     "[stats] prompt %d tok | prefill %.2f tok/s | decode %.2f tok/s (%d tok) | peak mem %.0f MB | "
                     "%s, %d thread(s), kv-cache %s\n",
                     n_prompt, n_prompt / t_prefill, decode_steps > 0 ? decode_steps / t_decode : 0.0, generated,
                     peak_rss_mb(), args.quant.c_str(), args.threads, args.kv_cache ? "on" : "off");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
