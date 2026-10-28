// CLI: ./engine --model m.bin --tokenizer tokenizer.json --prompt "..." [options]
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
    std::string perplexity;   // evaluate perplexity on this text file instead of generating
    std::string kernel = "blocked";
    int ppl_tokens = 0;       // max tokens to score in perplexity mode (0 = all)
    int max_tokens = 256;
    int threads = 1;
    int ctx = 0;  // 0 = prompt + max_tokens
    std::string quant;  // optional check that the model file has this format
    bool kv_cache = true;
    bool stop_at_eos = true;
    bool quiet = false;
    bool no_mmap = false;
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
        "  --threads N        threads for matmuls and attention (default 1)\n"
        "  --kernel K         naive | simd | blocked (default blocked)\n"
        "  --quant Q          fp32 | int8: assert the model file's weight format\n"
        "  --ctx N            context length to allocate (default prompt + max-tokens)\n"
        "  --no-kv-cache      naive baseline: recompute the whole sequence every token\n"
        "  --ignore-eos       keep generating past the end-of-text token\n"
        "  --tokens 1,2,3     feed raw token ids instead of --prompt\n"
        "  --dump-logits F    write the logits after the prompt to F (float32)\n"
        "  --ids-out F        write prompt + generated token ids to F\n"
        "  --quiet            don't stream generated text\n"
        "  --no-mmap          read weights into (huge-page) memory instead of mmap\n"
        "  --perplexity F     score text file F (chunks of --ctx tokens, default 512)\n"
        "  --ppl-tokens N     score at most N tokens in perplexity mode\n");
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
        else if (k == "--no-mmap") a.no_mmap = true;
        else if (k == "--kernel") a.kernel = val();
        else if (k == "--perplexity") a.perplexity = val();
        else if (k == "--ppl-tokens") a.ppl_tokens = std::stoi(val());
        else usage();
    }
    if (a.model.empty() || (a.prompt.empty() && a.tokens.empty() && a.perplexity.empty())) usage();
    if (!a.perplexity.empty() && a.tokenizer.empty()) {
        std::fprintf(stderr, "--perplexity needs --tokenizer\n");
        std::exit(1);
    }
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

kernels::Impl parse_kernel(const std::string& k) {
    if (k == "naive") return kernels::Impl::Naive;
    if (k == "simd") return kernels::Impl::Simd;
    if (k == "blocked") return kernels::Impl::Blocked;
    throw std::runtime_error("--kernel must be naive, simd or blocked");
}

// Perplexity = exp(mean negative log-likelihood of each next token), scored over
// non-overlapping chunks of `ctx` tokens (each chunk starts with an empty KV cache).
int run_perplexity(Model& model, const Tokenizer& tok, const Args& args) {
    std::ifstream f(args.perplexity, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + args.perplexity);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<int> ids = tok.encode(ss.str(), /*allow_special=*/false);
    if (args.ppl_tokens > 0 && int(ids.size()) > args.ppl_tokens) ids.resize(args.ppl_tokens);
    const int ctx = model.ctx_len(), V = model.config().vocab_size;
    double nll = 0.0;
    long scored = 0;
    double t0 = now_s();
    for (size_t start = 0; start + 1 < ids.size(); start += ctx) {
        const int n = int(std::min<size_t>(ctx, ids.size() - start));
        for (int p = 0; p + 1 < n; ++p) {
            const float* logits = model.forward(ids[start + p], p, true);
            float mx = *std::max_element(logits, logits + V);
            double sum = 0.0;
            for (int i = 0; i < V; ++i) sum += std::exp(double(logits[i] - mx));
            nll += -(double(logits[ids[start + p + 1]] - mx) - std::log(sum));
            ++scored;
        }
        std::fprintf(stderr, "\r  scored %ld tokens, running ppl %.4f", scored, std::exp(nll / scored));
    }
    std::fprintf(stderr, "\n");
    std::printf("perplexity %.4f over %ld tokens (ctx %d) in %.1fs\n", std::exp(nll / scored), scored, ctx,
                now_s() - t0);
    return 0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    try {
        ModelOptions opts;
        opts.threads = args.threads;
        opts.impl = parse_kernel(args.kernel);
        opts.no_mmap = args.no_mmap;

        std::unique_ptr<Tokenizer> tok;
        if (!args.tokenizer.empty()) tok = std::make_unique<Tokenizer>(args.tokenizer);

        if (!args.perplexity.empty()) {
            Model model(args.model, args.ctx > 0 ? args.ctx : 512, opts);
            return run_perplexity(model, *tok, args);
        }

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
        Model model(args.model, ctx, opts);
        const Config& cfg = model.config();
        const char* format = cfg.q8() ? "int8" : "fp32";
        if (!args.quant.empty() && args.quant != format)
            throw std::runtime_error("--quant " + args.quant + " but the model file is " + format +
                                     " (convert with scripts/convert.py --quant int8)");
        if (n_prompt + args.max_tokens > model.ctx_len())
            args.max_tokens = std::max(0, model.ctx_len() - n_prompt);
        Sampler sampler(args.sampler);

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
            if (args.stop_at_eos && tok && tok->is_stop(next)) break;
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
                     "%s, %s kernel (%s), %d thread(s), kv-cache %s\n",
                     n_prompt, n_prompt / t_prefill, decode_steps > 0 ? decode_steps / t_decode : 0.0, generated,
                     peak_rss_mb(), format, args.kernel.c_str(), kernels::simd_name(), args.threads,
                     args.kv_cache ? "on" : "off");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
