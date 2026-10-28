// Byte-level BPE tokenizer (GPT-2 style), loaded from a Hugging Face tokenizer.json.
//
// Encoding pipeline, mirroring the HF `tokenizers` library:
//   1. split out added/special tokens (e.g. "<|im_start|>") by exact match
//   2. pre-tokenize the remaining text: optional Digits split, then the GPT-2 regex
//   3. map every UTF-8 byte of each piece to a printable "byte-level" character
//   4. apply BPE merges by rank until no mergeable pair remains, then look up ids
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path);

    // allow_special: recognise added/special tokens written literally in the text.
    std::vector<int> encode(const std::string& text, bool allow_special = true) const;
    std::string decode(const std::vector<int>& ids) const;
    // Decode a single token to raw bytes (may be a partial UTF-8 sequence while streaming).
    std::string decode_token(int id) const;

    int vocab_size() const { return int(id_to_token_.size()); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    struct AddedToken {
        std::string content;
        int id;
        bool special;
    };

    std::unordered_map<std::string, int> vocab_;       // byte-level string -> id
    std::vector<std::string> id_to_token_;             // id -> byte-level string (or literal for added)
    std::vector<bool> is_added_;                        // id is an added token (decoded literally)
    std::unordered_map<std::string, int> merge_rank_;   // "left right" -> rank
    std::vector<AddedToken> added_;                     // sorted longest-first for matching

    bool split_digits_ = false;  // Digits pre-tokenizer with individual_digits
    bool use_regex_ = true;      // ByteLevel pre-tokenizer applies the GPT-2 regex
    bool add_prefix_space_ = false;
    int bos_id_ = -1, eos_id_ = -1;

    std::string byte_to_char_[256];                     // byte -> UTF-8 of its byte-level char
    std::unordered_map<uint32_t, uint8_t> char_to_byte_;

    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    void encode_ordinary(const std::string& text, std::vector<int>& out) const;
    void bpe(const std::string& piece, std::vector<int>& out) const;
};
