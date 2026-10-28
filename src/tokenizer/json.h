// Minimal JSON parser: just enough to read a Hugging Face tokenizer.json.
// Recursive descent into a tree of Value nodes; strings are decoded to UTF-8.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    // Objects keep insertion order (tokenizer.json vocab order is not needed, but keeps output stable).
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return type == Type::Null; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    // Object lookup; returns nullptr if missing or not an object.
    const Value* find(std::string_view key) const {
        if (type != Type::Object) return nullptr;
        for (auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    const Value& at(std::string_view key) const {
        const Value* v = find(key);
        if (!v) throw std::runtime_error("json: missing key '" + std::string(key) + "'");
        return *v;
    }
    const std::string& as_string() const {
        if (type != Type::String) throw std::runtime_error("json: expected string");
        return str;
    }
    double as_number() const {
        if (type != Type::Number) throw std::runtime_error("json: expected number");
        return num;
    }
    bool as_bool() const {
        if (type != Type::Bool) throw std::runtime_error("json: expected bool");
        return b;
    }
};

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    Value parse() {
        Value v = value();
        ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    std::string_view s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const char* msg) {
        throw std::runtime_error(std::string("json parse error at byte ") + std::to_string(i_) + ": " + msg);
    }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    char peek() {
        if (i_ >= s_.size()) fail("unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++i_;
    }
    bool consume_literal(std::string_view lit) {
        if (s_.substr(i_, lit.size()) == lit) {
            i_ += lit.size();
            return true;
        }
        return false;
    }

    Value value() {
        ws();
        Value v;
        char c = peek();
        if (c == '{') {
            v.type = Value::Type::Object;
            ++i_;
            ws();
            if (peek() == '}') { ++i_; return v; }
            for (;;) {
                ws();
                std::string key = string();
                ws();
                expect(':');
                v.obj.emplace_back(std::move(key), value());
                ws();
                if (peek() == ',') { ++i_; continue; }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            v.type = Value::Type::Array;
            ++i_;
            ws();
            if (peek() == ']') { ++i_; return v; }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (peek() == ',') { ++i_; continue; }
                expect(']');
                return v;
            }
        }
        if (c == '"') {
            v.type = Value::Type::String;
            v.str = string();
            return v;
        }
        if (consume_literal("true")) { v.type = Value::Type::Bool; v.b = true; return v; }
        if (consume_literal("false")) { v.type = Value::Type::Bool; v.b = false; return v; }
        if (consume_literal("null")) return v;
        // number
        size_t start = i_;
        while (i_ < s_.size() && (std::string_view("+-0123456789.eE").find(s_[i_]) != std::string_view::npos)) ++i_;
        if (start == i_) fail("unexpected character");
        v.type = Value::Type::Number;
        v.num = std::stod(std::string(s_.substr(start, i_ - start)));
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i_ + 4 > s_.size()) fail("bad \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // surrogate pair
                        if (!consume_literal("\\u")) fail("lone high surrogate");
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
};

inline Value parse(std::string_view s) { return Parser(s).parse(); }

}  // namespace json
