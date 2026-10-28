// OpenAI-compatible completions server on raw POSIX sockets (no dependencies).
//
//   ./build/server --model m.bin --tokenizer tokenizer.json [--port 8080] [--threads 4] [--ctx 2048]
//
//   POST /v1/completions  {"prompt": "...", "max_tokens": 64, "temperature": 0.7, "top_p": 0.9,
//                          "top_k": 40, "seed": 1, "stream": true, "stop": ["\n\n"]}
//   GET  /v1/models, GET /health
//
// With "stream": true the reply is Server-Sent Events: one `data: {...}` chunk per token and a
// final `data: [DONE]`, the same wire format as the OpenAI API, so its client libraries work.
// Requests are served one at a time: a single model instance and KV cache, batch size 1.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model.h"
#include "sampler.h"
#include "tokenizer/json.h"
#include "tokenizer/tokenizer.h"

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += char(c);
                }
        }
    }
    return out;
}

// Length of the longest prefix of s that ends on a complete UTF-8 character. A token can end in
// the middle of a multi-byte character; the tail is held back until the next token completes it.
size_t complete_utf8_prefix(const std::string& s) {
    size_t i = s.size();
    for (int back = 1; back <= 4 && i > 0; ++back) {
        const unsigned char c = s[--i];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep walking back to the lead byte
        const int need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        return need <= back ? s.size() : i;  // last character complete, or cut before it
    }
    return s.size();
}

bool send_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = send(fd, data.data() + off, data.size() - off, 0);
        if (n <= 0) return false;
        off += size_t(n);
    }
    return true;
}

void send_response(int fd, int status, const std::string& body, const char* type = "application/json") {
    const char* reason = status == 200 ? "OK" : status == 404 ? "Not Found" : status == 400 ? "Bad Request" : "Error";
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\nContent-Type: " + type +
                       "\r\nContent-Length: " + std::to_string(body.size()) +
                       "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    send_all(fd, head + body);
}

void send_error(int fd, int status, const std::string& msg) {
    send_response(fd, status, "{\"error\":{\"message\":\"" + json_escape(msg) + "\",\"type\":\"invalid_request_error\"}}");
}

struct Request {
    std::string method, path, body;
};

// Reads one HTTP/1.1 request (headers, then Content-Length bytes of body).
bool read_request(int fd, Request& req) {
    std::string buf;
    char chunk[8192];
    size_t header_end;
    while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0 || buf.size() > (1 << 20)) return false;
        buf.append(chunk, size_t(n));
    }
    const std::string head = buf.substr(0, header_end);
    const size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = head.substr(0, sp1);
    req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t content_length = 0;
    std::string lower = head;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (size_t p = lower.find("\r\ncontent-length:"); p != std::string::npos)
        content_length = std::stoul(head.substr(p + 17));
    if (content_length > (16u << 20)) return false;
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_length) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        req.body.append(chunk, size_t(n));
    }
    req.body.resize(content_length);
    return true;
}

struct Server {
    Model& model;
    Tokenizer& tok;
    std::string model_name;
    long next_id = 1;

    void handle(int fd, const Request& req) {
        if (req.method == "GET" && req.path == "/health") return send_response(fd, 200, "{\"status\":\"ok\"}");
        if (req.method == "GET" && req.path == "/v1/models")
            return send_response(fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"" + json_escape(model_name) +
                                              "\",\"object\":\"model\",\"owned_by\":\"local\"}]}");
        if (req.method == "OPTIONS") return send_response(fd, 200, "");
        if (req.method != "POST" || req.path != "/v1/completions") return send_error(fd, 404, "unknown endpoint");
        try {
            completion(fd, json::parse(req.body));
        } catch (const std::exception& e) {
            send_error(fd, 400, e.what());
        }
    }

    void completion(int fd, const json::Value& body) {
        const json::Value* p = body.find("prompt");
        if (!p || !p->is_string()) throw std::runtime_error("'prompt' must be a string");
        auto num = [&](const char* k, double dflt) {
            const json::Value* v = body.find(k);
            return v && !v->is_null() ? v->as_number() : dflt;
        };
        int max_tokens = int(num("max_tokens", 16));
        SamplerConfig sc;
        sc.temperature = float(num("temperature", 1.0));
        sc.top_p = float(num("top_p", 1.0));
        sc.top_k = int(num("top_k", 0));
        sc.seed = uint64_t(num("seed", double(std::time(nullptr))));
        const json::Value* st = body.find("stream");
        const bool stream = st && st->type == json::Value::Type::Bool && st->as_bool();
        std::vector<std::string> stops;
        if (const json::Value* s = body.find("stop")) {
            if (s->is_string()) stops.push_back(s->str);
            else if (s->is_array())
                for (auto& v : s->arr) stops.push_back(v.as_string());
        }

        std::vector<int> ids = tok.encode(p->str);
        if (ids.empty()) throw std::runtime_error("prompt is empty");
        if (int(ids.size()) >= model.ctx_len())
            throw std::runtime_error("prompt is " + std::to_string(ids.size()) + " tokens; the context is " +
                                     std::to_string(model.ctx_len()));
        max_tokens = std::max(0, std::min(max_tokens, model.ctx_len() - int(ids.size())));

        const std::string id = "cmpl-" + std::to_string(next_id++);
        const long created = long(std::time(nullptr));
        auto chunk_json = [&](const std::string& text, const char* finish) {
            return "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
                   ",\"model\":\"" + json_escape(model_name) + "\",\"choices\":[{\"text\":\"" + json_escape(text) +
                   "\",\"index\":0,\"logprobs\":null,\"finish_reason\":" +
                   (finish ? "\"" + std::string(finish) + "\"" : "null") + "}]";
        };
        if (stream &&
            !send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                          "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"))
            return;

        const auto t0 = std::chrono::steady_clock::now();
        const float* logits = nullptr;
        for (size_t i = 0; i < ids.size(); ++i) logits = model.forward(ids[i], int(i), i + 1 == ids.size());

        Sampler sampler(sc);
        std::string text, pending;  // pending: bytes not yet sent (partial UTF-8 / possible stop prefix)
        const char* finish = "length";
        int generated = 0;
        int pos = int(ids.size());
        for (; generated < max_tokens; ++generated) {
            const int next = sampler.sample(logits, model.config().vocab_size);
            if (tok.is_stop(next)) {
                finish = "stop";
                break;
            }
            pending += tok.decode_token(next);
            // Stop sequences: cut the text at the first match and finish.
            bool stopped = false;
            for (const auto& s : stops) {
                if (s.empty()) continue;
                if (size_t at = (text + pending).find(s); at != std::string::npos) {
                    pending = (text + pending).substr(text.size(), at - text.size());
                    stopped = true;
                    break;
                }
            }
            // Hold back a tail that could still grow into a stop sequence, and incomplete UTF-8.
            size_t safe = complete_utf8_prefix(pending);
            if (!stopped)
                for (const auto& s : stops)
                    for (size_t k = std::min(s.size() - 1, pending.size()); k > 0; --k)
                        if (pending.compare(pending.size() - k, k, s, 0, k) == 0) {
                            safe = std::min(safe, pending.size() - k);
                            break;
                        }
            if (safe > 0) {
                const std::string out = pending.substr(0, safe);
                text += out;
                pending.erase(0, safe);
                if (stream && !send_all(fd, "data: " + chunk_json(out, nullptr) + "}\n\n")) return;  // client left
            }
            if (stopped) {
                finish = "stop";
                ++generated;
                break;
            }
            if (generated + 1 < max_tokens) logits = model.forward(next, pos++, true);
        }
        text += pending;  // flush anything held back
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::string usage = ",\"usage\":{\"prompt_tokens\":" + std::to_string(ids.size()) +
                                  ",\"completion_tokens\":" + std::to_string(generated) +
                                  ",\"total_tokens\":" + std::to_string(ids.size() + generated) + "}";
        std::fprintf(stderr, "[%s] %zu prompt + %d completion tokens in %.2fs (%s)\n", id.c_str(), ids.size(), generated,
                     secs, stream ? "stream" : "json");
        if (stream) {
            send_all(fd, "data: " + chunk_json(pending, finish) + usage + "}\n\ndata: [DONE]\n\n");
        } else {
            send_response(fd, 200, chunk_json(text, finish) + usage + "}");
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tok_path;
    int port = 8080, threads = 1, ctx = 2048;
    kernels::Impl impl = kernels::Impl::Blocked;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model_path = v;
        else if (k == "--tokenizer") tok_path = v;
        else if (k == "--port") port = std::stoi(v);
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--ctx") ctx = std::stoi(v);
        else if (k == "--kernel") impl = v == "naive" ? kernels::Impl::Naive : v == "simd" ? kernels::Impl::Simd
                                                                                          : kernels::Impl::Blocked;
        else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return 1;
        }
    }
    if (model_path.empty() || tok_path.empty()) {
        std::fprintf(stderr,
                     "usage: server --model m.bin --tokenizer tokenizer.json [--port 8080] [--threads N] [--ctx 2048]\n");
        return 1;
    }
    signal(SIGPIPE, SIG_IGN);  // a client hanging up mid-stream must not kill the server

    try {
        Tokenizer tok(tok_path);
        ModelOptions opts;
        opts.threads = threads;
        opts.impl = impl;
        Model model(model_path, ctx, opts);
        // Report the model's directory name (models/SmolLM2-360M/model.bin -> "SmolLM2-360M").
        std::string name = model_path;
        if (size_t slash = name.find_last_of('/'); slash != std::string::npos) {
            name.resize(slash);
            if (size_t prev = name.find_last_of('/'); prev != std::string::npos) name = name.substr(prev + 1);
        }
        Server server{model, tok, name};

        int sfd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // local only
        addr.sin_port = htons(uint16_t(port));
        if (bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || listen(sfd, 16) < 0)
            throw std::runtime_error("cannot listen on port " + std::to_string(port));
        std::fprintf(stderr, "serving %s on http://127.0.0.1:%d/v1/completions (%d threads)\n", name.c_str(), port,
                     threads);
        for (;;) {
            int fd = accept(sfd, nullptr, nullptr);
            if (fd < 0) continue;
            Request req;
            if (read_request(fd, req)) server.handle(fd, req);
            close(fd);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
