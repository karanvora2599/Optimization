#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

// ─── TextDataset ──────────────────────────────────────────────────────────────
// Reads a plain-text file, tokenizes it with BPETokenizer, and packs tokens
// into fixed-length context windows (block_size tokens each).
// Each sample is a pair {input_ids, labels} where labels = input_ids shifted
// left by one position (standard language modeling objective).
class BPETokenizer;  // forward decl

struct TextSample {
    torch::Tensor input_ids;  // [block_size]
    torch::Tensor labels;     // [block_size]  (input_ids shifted by 1)
};

class TextDataset {
public:
    TextDataset(const std::string& text_path,
                const BPETokenizer& tokenizer,
                int64_t             block_size);

    // Number of samples (non-overlapping windows)
    size_t size() const { return samples_.size(); }

    // Get the i-th sample
    TextSample get(size_t idx) const { return samples_[idx]; }

    // Build a batch from a list of indices: returns {input_ids [B,T], labels [B,T]}
    std::pair<torch::Tensor, torch::Tensor>
    make_batch(const std::vector<size_t>& indices) const;

private:
    std::vector<TextSample> samples_;
};
