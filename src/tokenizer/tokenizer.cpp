#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "tokenizer/json.h"
#include "tokenizer/unicode_tables.h"

namespace {

// ---- Unicode helpers ---------------------------------------------------------------

template <size_t N>
bool in_ranges(const unicode_tables::CodepointRange (&rs)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {  // binary search over sorted, disjoint ranges
        size_t mid = (lo + hi) / 2;
        if (cp < rs[mid].lo) hi = mid;
        else if (cp > rs[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return in_ranges(unicode_tables::kLetters, cp);
}

bool is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(unicode_tables::kNumbers, cp);
}

// Unicode White_Space property (what \s means in the Rust regex the HF tokenizers use).
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

struct Cp {
    uint32_t cp;
    size_t byte;  // byte offset where this codepoint starts
};

std::vector<Cp> decode_utf8(const std::string& s) {
    std::vector<Cp> out;
    out.reserve(s.size() + 1);
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { cp = 0xFFFD; len = 1; }  // invalid lead byte: treat as one "other" char
        if (i + len > s.size()) { cp = 0xFFFD; len = 1; }
        for (int k = 1; k < len; ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
        out.push_back({cp, i});
        i += len;
    }
    out.push_back({0, s.size()});  // sentinel holding the end offset
    return out;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
}

// ---- GPT-2 pre-tokenizer -------------------------------------------------------------
// Hand-written equivalent of:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Alternatives are tried in order at each position, exactly like the regex engine.
void gpt2_split(const std::string& text, std::vector<std::string>& pieces) {
    std::vector<Cp> cps = decode_utf8(text);
    const size_t n = cps.size() - 1;
    auto other = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        uint32_t c = cps[i].cp;
        uint32_t next = i + 1 < n ? cps[i + 1].cp : 0;
        bool has_next = i + 1 < n;

        if (c == '\'' && has_next) {  // contractions (case-sensitive, as in GPT-2)
            uint32_t n2 = i + 2 < n ? cps[i + 2].cp : 0;
            if (next == 's' || next == 't' || next == 'm' || next == 'd') j = i + 2;
            else if ((next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l')) j = i + 3;
        }
        if (j == i) {
            // " ?X+" for X in {letters, numbers, other}
            size_t k = (c == ' ' && has_next) ? i + 1 : i;
            uint32_t ck = cps[k].cp;
            bool (*cls)(uint32_t) = nullptr;
            if (k < n) {
                if (is_letter(ck)) cls = is_letter;
                else if (is_number(ck)) cls = is_number;
                else if (other(ck)) cls = +[](uint32_t x) { return !is_space(x) && !is_letter(x) && !is_number(x); };
            }
            if (cls) {
                j = k;
                while (j < n && cls(cps[j].cp)) ++j;
            }
        }
        if (j == i) {
            // whitespace: \s+(?!\S) then \s+
            size_t k = i;
            while (k < n && is_space(cps[k].cp)) ++k;
            if (k == n) j = k;                 // run reaches the end
            else if (k - i > 1) j = k - 1;     // leave the last space to attach to the next word
            else j = k;                        // single whitespace before a non-space
        }
        if (j == i) j = i + 1;  // unreachable in practice; guarantees progress
        pieces.emplace_back(text.substr(cps[i].byte, cps[j].byte - cps[i].byte));
        i = j;
    }
}

// Digits pre-tokenizer (individual_digits=true): every numeric char becomes its own piece.
void digits_split(const std::string& text, std::vector<std::string>& pieces) {
    std::vector<Cp> cps = decode_utf8(text);
    const size_t n = cps.size() - 1;
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
        if (is_number(cps[i].cp)) {
            if (i > start) pieces.emplace_back(text.substr(cps[start].byte, cps[i].byte - cps[start].byte));
            pieces.emplace_back(text.substr(cps[i].byte, cps[i + 1].byte - cps[i].byte));
            start = i + 1;
        }
    }
    if (start < n) pieces.emplace_back(text.substr(cps[start].byte));
}

const char* kGpt2Pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

}  // namespace

// ---- Loading -------------------------------------------------------------------------

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open tokenizer file " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value root = json::parse(ss.str());

    // GPT-2 byte <-> printable char mapping. Printable Latin-1 bytes map to themselves;
    // the rest are shifted to codepoints 256+ so every byte has a visible character.
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = printable ? uint32_t(b) : uint32_t(256 + extra++);
        append_utf8(byte_to_char_[b], cp);
        char_to_byte_[cp] = uint8_t(b);
    }

    const json::Value& model = root.at("model");
    if (model.at("type").as_string() != "BPE") throw std::runtime_error("only BPE tokenizers are supported");

    const json::Value& vocab = model.at("vocab");
    int max_id = -1;
    for (auto& [tok, id] : vocab.obj) max_id = std::max(max_id, int(id.as_number()));
    const json::Value* added = root.find("added_tokens");
    if (added)
        for (auto& t : added->arr) max_id = std::max(max_id, int(t.at("id").as_number()));
    id_to_token_.resize(max_id + 1);
    is_added_.assign(max_id + 1, false);
    for (auto& [tok, id] : vocab.obj) {
        vocab_[tok] = int(id.as_number());
        id_to_token_[int(id.as_number())] = tok;
    }

    const json::Value& merges = model.at("merges");
    for (size_t r = 0; r < merges.arr.size(); ++r) {
        const json::Value& m = merges.arr[r];
        std::string key;
        if (m.is_string()) key = m.str;  // older format: "a b"
        else key = m.arr.at(0).as_string() + " " + m.arr.at(1).as_string();  // newer: ["a", "b"]
        merge_rank_.emplace(key, int(r));
    }

    if (added) {
        for (auto& t : added->arr) {
            AddedToken a{t.at("content").as_string(), int(t.at("id").as_number()), t.at("special").as_bool()};
            id_to_token_[a.id] = a.content;
            is_added_[a.id] = true;
            added_.push_back(a);
        }
        std::sort(added_.begin(), added_.end(),
                  [](const AddedToken& a, const AddedToken& b) { return a.content.size() > b.content.size(); });
    }
    for (const char* name : {"<|endoftext|>", "</s>", "<|end_of_text|>"}) {
        auto it = std::find_if(added_.begin(), added_.end(), [&](const AddedToken& a) { return a.content == name; });
        if (it != added_.end()) { eos_id_ = it->id; break; }
    }
    for (const char* name : {"<|begin_of_text|>", "<s>", "<|im_start|>"}) {
        auto it = std::find_if(added_.begin(), added_.end(), [&](const AddedToken& a) { return a.content == name; });
        if (it != added_.end()) { bos_id_ = it->id; break; }
    }

    // Pre-tokenizer: support ByteLevel, Digits, and a Sequence of them (covers SmolLM2 / GPT-2).
    if (const json::Value* norm = root.find("normalizer"); norm && !norm->is_null())
        throw std::runtime_error("tokenizer normalizers are not supported");
    std::vector<const json::Value*> pres;
    if (const json::Value* pre = root.find("pre_tokenizer"); pre && !pre->is_null()) {
        if (pre->at("type").as_string() == "Sequence")
            for (auto& p : pre->at("pretokenizers").arr) pres.push_back(&p);
        else
            pres.push_back(pre);
    }
    use_regex_ = false;
    for (const json::Value* p : pres) {
        const std::string& type = p->at("type").as_string();
        if (type == "ByteLevel") {
            const json::Value* ur = p->find("use_regex");
            use_regex_ = use_regex_ || !ur || ur->as_bool();
            if (const json::Value* aps = p->find("add_prefix_space")) add_prefix_space_ = aps->as_bool();
        } else if (type == "Digits") {
            split_digits_ = p->at("individual_digits").as_bool();
        } else if (type == "Split" && p->at("pattern").find("Regex") &&
                   p->at("pattern").at("Regex").as_string() == kGpt2Pattern) {
            use_regex_ = true;
        } else {
            throw std::runtime_error("unsupported pre_tokenizer type: " + type);
        }
    }
}

// ---- Encoding ------------------------------------------------------------------------

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    if (auto it = cache_.find(piece); it != cache_.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
    }
    // Start from one symbol per byte (as its byte-level character).
    std::vector<std::string> syms;
    syms.reserve(piece.size());
    for (unsigned char b : piece) syms.push_back(byte_to_char_[b]);

    // Repeatedly merge the adjacent pair with the lowest merge rank.
    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        size_t best = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == INT_MAX) break;
        // Merge every occurrence of this pair, left to right (same as HF).
        std::string left = syms[best], right = syms[best + 1];
        std::vector<std::string> merged;
        merged.reserve(syms.size());
        for (size_t i = 0; i < syms.size(); ++i) {
            if (i + 1 < syms.size() && syms[i] == left && syms[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(std::move(syms[i]));
            }
        }
        syms.swap(merged);
    }

    std::vector<int> ids;
    for (auto& s : syms) {
        auto it = vocab_.find(s);
        if (it == vocab_.end()) throw std::runtime_error("BPE produced a symbol missing from the vocab");
        ids.push_back(it->second);
    }
    out.insert(out.end(), ids.begin(), ids.end());
    if (cache_.size() < 100000) cache_.emplace(piece, std::move(ids));
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    std::vector<std::string> stage = {add_prefix_space_ && text[0] != ' ' ? " " + text : text};
    if (split_digits_) {
        std::vector<std::string> next;
        for (auto& s : stage) digits_split(s, next);
        stage.swap(next);
    }
    if (use_regex_) {
        std::vector<std::string> next;
        for (auto& s : stage) gpt2_split(s, next);
        stage.swap(next);
    }
    for (auto& piece : stage) bpe(piece, out);
}

std::vector<int> Tokenizer::encode(const std::string& text, bool allow_special) const {
    std::vector<int> out;
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const AddedToken* hit = nullptr;
        for (auto& a : added_) {
            if (a.special && !allow_special) continue;
            if (text.compare(i, a.content.size(), a.content) == 0) { hit = &a; break; }
        }
        if (hit) {
            encode_ordinary(text.substr(start, i - start), out);
            out.push_back(hit->id);
            i += hit->content.size();
            start = i;
        } else {
            ++i;
        }
    }
    encode_ordinary(text.substr(start), out);
    return out;
}

// ---- Decoding ------------------------------------------------------------------------

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= int(id_to_token_.size())) return "";
    const std::string& tok = id_to_token_[id];
    if (is_added_[id]) return tok;
    std::string out;
    for (const Cp& c : decode_utf8(tok)) {
        if (c.byte == tok.size()) break;  // sentinel
        auto it = char_to_byte_.find(c.cp);
        if (it != char_to_byte_.end()) out += char(it->second);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode_token(id);
    return out;
}
