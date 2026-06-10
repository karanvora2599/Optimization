#pragma once
#include <cstdint>

// GPT-2 Small (124M) defaults. Change n_layer/n_head/n_embd for Medium/Large.
struct GPT2Config {
    int64_t vocab_size  = 50257;  // GPT-2 standard vocabulary
    int64_t block_size  = 1024;   // maximum sequence length
    int64_t n_embd      = 768;    // embedding / hidden dimension
    int64_t n_layer     = 12;     // number of transformer blocks
    int64_t n_head      = 12;     // number of attention heads
    float   dropout     = 0.1f;   // dropout probability (set 0 for inference)
};

// LoRA hyper-parameters
struct LoRAConfig {
    int64_t rank   = 8;     // low-rank dimension
    float   alpha  = 16.0f; // scaling factor  (effective lr = alpha / rank)
};
