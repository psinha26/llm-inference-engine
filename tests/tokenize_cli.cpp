// Test helper: tokenize every string in a JSON array and print one line of ids per string.
//   ./tokenize tokenizer.json strings.json > ids.txt
// tests/check_tokenizer.py compares the output against the Hugging Face tokenizer.
#include <fstream>
#include <iostream>
#include <sstream>

#include "tokenizer/json.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tokenize tokenizer.json strings.json\n";
        return 1;
    }
    Tokenizer tok(argv[1]);
    std::ifstream f(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value strings = json::parse(ss.str());
    int roundtrip_fail = 0;
    for (const auto& s : strings.arr) {
        std::vector<int> ids = tok.encode(s.as_string());
        for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i];
        std::cout << "\n";
        if (tok.decode(ids) != s.as_string()) ++roundtrip_fail;
    }
    std::cerr << "decode(encode(s)) != s for " << roundtrip_fail << " of " << strings.arr.size() << " strings\n";
    return 0;
}
