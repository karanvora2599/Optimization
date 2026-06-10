#include "tokenizer/bpe_tokenizer.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>

using json = nlohmann::json;

// ─── Byte-level unicode map (mirrors GPT-2's bytes_to_unicode()) ──────────────
static std::unordered_map<uint8_t, std::string> build_byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> bs;
    // Printable ASCII ranges that map to themselves
    for (int b = '!'; b <= '~'; ++b)  bs[b] = std::string(1, (char)b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs[b] = std::string(1, (char)b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs[b] = std::string(1, (char)b);
    // Remaining bytes get mapped to unicode code points starting at U+0100
    int n = 256;
    for (int b = 0; b < 256; ++b) {
        if (bs.find((uint8_t)b) == bs.end()) {
            // Encode as UTF-8 code point (n++)
            char buf[5];
            int cp = n++;
            if (cp < 0x80) {
                buf[0] = (char)cp; buf[1] = 0;
            } else if (cp < 0x800) {
                buf[0] = (char)(0xC0 | (cp >> 6));
                buf[1] = (char)(0x80 | (cp & 0x3F));
                buf[2] = 0;
            } else {
                buf[0] = (char)(0xE0 | (cp >> 12));
                buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                buf[2] = (char)(0x80 | (cp & 0x3F));
                buf[3] = 0;
            }
            bs[(uint8_t)b] = std::string(buf);
        }
    }
    return bs;
}

// Convert a raw UTF-8 string to the GPT-2 byte representation
static std::string bytes_to_gpt2(const std::string& s,
    const std::unordered_map<uint8_t, std::string>& b2u)
{
    std::string out;
    for (unsigned char c : s) {
        auto it = b2u.find(c);
        if (it != b2u.end()) out += it->second;
    }
    return out;
}

// ─── BPETokenizer::load ───────────────────────────────────────────────────────
bool BPETokenizer::load(const std::string& vocab_path,
                        const std::string& merges_path) {
    // Build byte->unicode mapping
    byte_to_unicode_ = build_byte_to_unicode();
    for (auto& [b, u] : byte_to_unicode_)
        unicode_to_byte_[u] = b;

    // Load vocab.json
    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            std::cerr << "[Tokenizer] Cannot open vocab: " << vocab_path << "\n";
            return false;
        }
        json vocab_json;
        f >> vocab_json;
        for (auto& [token, id] : vocab_json.items()) {
            int64_t tid = id.get<int64_t>();
            encoder_[token] = tid;
            decoder_[tid]   = token;
        }
    }

    // Load merges.txt (skip header line)
    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            std::cerr << "[Tokenizer] Cannot open merges: " << merges_path << "\n";
            return false;
        }
        std::string line;
        bool first = true;
        int rank = 0;
        while (std::getline(f, line)) {
            if (first) { first = false; continue; }  // skip #version line
            if (line.empty()) continue;
            auto sp = line.find(' ');
            if (sp == std::string::npos) continue;
            bpe_ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
        }
    }

    std::cout << "[Tokenizer] Loaded " << encoder_.size()
              << " tokens and " << bpe_ranks_.size() << " merges.\n";
    return true;
}

// ─── BPE core ────────────────────────────────────────────────────────────────
static std::vector<std::string> get_pairs(const std::vector<std::string>& word) {
    std::vector<std::string> pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i)
        pairs.push_back(word[i] + " " + word[i+1]);
    return pairs;
}

std::vector<std::string>
BPETokenizer::bpe(const std::string& token) const {
    auto it = cache_.find(token);
    if (it != cache_.end()) return it->second;

    std::vector<std::string> word;
    for (char c : token) word.push_back(std::string(1, c));

    while (word.size() > 1) {
        // Find the merge with the lowest rank
        int best_rank = INT_MAX;
        std::pair<std::string,std::string> best_pair;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto key = std::make_pair(word[i], word[i+1]);
            auto rit = bpe_ranks_.find(key);
            if (rit != bpe_ranks_.end() && rit->second < best_rank) {
                best_rank = rit->second;
                best_pair = key;
            }
        }
        if (best_rank == INT_MAX) break;

        // Apply merge
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() &&
                word[i] == best_pair.first && word[i+1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_word.push_back(word[i++]);
            }
        }
        word = std::move(new_word);
    }

    cache_[token] = word;
    return word;
}

// ─── BPETokenizer::encode ─────────────────────────────────────────────────────
std::vector<int64_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids;

    // Simple whitespace-aware pre-tokenisation (GPT-2 style: prepend space)
    std::istringstream iss(text);
    std::string word;
    bool first_word = true;
    while (iss >> word) {
        std::string prefixed = (first_word ? "" : " ") + word;
        first_word = false;

        // Convert each byte to its unicode representation
        std::string bpe_token = bytes_to_gpt2(prefixed, byte_to_unicode_);

        // Apply BPE merges
        auto pieces = bpe(bpe_token);
        for (const auto& piece : pieces) {
            auto eit = encoder_.find(piece);
            if (eit != encoder_.end())
                ids.push_back(eit->second);
        }
    }
    return ids;
}

// ─── BPETokenizer::decode ─────────────────────────────────────────────────────
std::string BPETokenizer::decode(const std::vector<int64_t>& tokens) const {
    std::string text;
    for (int64_t id : tokens) {
        auto it = decoder_.find(id);
        if (it == decoder_.end()) continue;
        for (const auto& ch : it->second) {
            auto uit = unicode_to_byte_.find(std::string(1, ch));
            if (uit != unicode_to_byte_.end())
                text += (char)uit->second;
            else
                text += ch;
        }
    }
    return text;
}
