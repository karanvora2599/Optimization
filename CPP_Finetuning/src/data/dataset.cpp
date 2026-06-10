#include "data/dataset.h"
#include "tokenizer/bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

TextDataset::TextDataset(const std::string& text_path,
                         const BPETokenizer& tokenizer,
                         int64_t             block_size)
{
    // Read entire file
    std::ifstream f(text_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open text file: " + text_path);

    std::ostringstream buf;
    buf << f.rdbuf();
    std::string text = buf.str();

    std::cout << "[Dataset] Tokenizing " << text.size() << " bytes…\n";

    // Tokenize the entire text at once
    auto all_tokens = tokenizer.encode(text);
    std::cout << "[Dataset] Got " << all_tokens.size() << " tokens.\n";

    // Build non-overlapping windows of size (block_size + 1)
    // +1 so we can form input[0..T-1] and label[1..T] from the same window
    int64_t window = block_size + 1;
    int64_t n_windows = static_cast<int64_t>(all_tokens.size()) / window;

    if (n_windows == 0)
        throw std::runtime_error(
            "Text is too short for block_size=" + std::to_string(block_size) +
            ". Found only " + std::to_string(all_tokens.size()) + " tokens.");

    samples_.reserve(n_windows);
    for (int64_t i = 0; i < n_windows; ++i) {
        auto begin = all_tokens.begin() + i * window;
        auto end   = begin + window;

        auto chunk = torch::tensor(
            std::vector<int64_t>(begin, end),
            torch::TensorOptions().dtype(torch::kLong));

        TextSample sample;
        sample.input_ids = chunk.slice(0, 0, block_size);   // [T]
        sample.labels    = chunk.slice(0, 1, block_size + 1); // [T]
        samples_.push_back(std::move(sample));
    }

    std::cout << "[Dataset] Created " << samples_.size() << " samples.\n";
}

std::pair<torch::Tensor, torch::Tensor>
TextDataset::make_batch(const std::vector<size_t>& indices) const {
    std::vector<torch::Tensor> inputs, labels;
    inputs.reserve(indices.size());
    labels.reserve(indices.size());

    for (size_t idx : indices) {
        inputs.push_back(samples_[idx].input_ids.unsqueeze(0));
        labels.push_back(samples_[idx].labels.unsqueeze(0));
    }

    return {torch::cat(inputs, 0), torch::cat(labels, 0)};
}
