#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>

// ─── GPT-2 Byte-Level BPE Tokenizer ──────────────────────────────────────────
// Loads vocab.json and merges.txt produced by scripts/export_gpt2.py.
// Implements the same byte-level BPE encoding as HuggingFace GPT2Tokenizer.
class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load vocab.json and merges.txt. Returns false on error.
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Encode a UTF-8 string to token ids
    std::vector<int64_t> encode(const std::string& text) const;

    // Decode token ids to UTF-8 string
    std::string decode(const std::vector<int64_t>& tokens) const;

    int64_t vocab_size()   const { return static_cast<int64_t>(encoder_.size()); }
    int64_t eos_token_id() const { return 50256; }

private:
    // Internal BPE application on a single (pre-tokenised) word string
    std::vector<std::string> bpe(const std::string& token) const;

    // token string -> id
    std::unordered_map<std::string, int64_t> encoder_;
    // id -> token string
    std::unordered_map<int64_t, std::string> decoder_;

    // BPE merge rules: pair of strings -> rank (lower = applied first)
    std::map<std::pair<std::string,std::string>, int> bpe_ranks_;

    // GPT-2 byte-level mappings
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Cache of already-computed BPE splits
    mutable std::unordered_map<std::string, std::vector<std::string>> cache_;
};
