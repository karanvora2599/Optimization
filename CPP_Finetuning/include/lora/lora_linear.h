#pragma once
#include <torch/torch.h>
#include "config.h"

// ─── LoRALinear ───────────────────────────────────────────────────────────────
// Replaces a frozen Linear layer W with  W + (alpha/rank) * B @ A.
// Only A and B are trainable; W is kept as a plain tensor (no gradient).
struct LoRALinearImpl : torch::nn::Module {
    LoRALinearImpl(int64_t in_features,
                   int64_t out_features,
                   const LoRAConfig& lora_cfg,
                   bool    has_bias = false);

    torch::Tensor forward(torch::Tensor x);

    // Frozen base weight (no grad)
    torch::Tensor weight;
    torch::Tensor base_bias;
    bool          has_bias;

    // Trainable low-rank matrices
    torch::nn::Linear lora_A{nullptr};  // in  → rank   (Gaussian init)
    torch::nn::Linear lora_B{nullptr};  // rank → out    (zero init)

    float scaling;  // alpha / rank
};
TORCH_MODULE(LoRALinear);

// Helper: convert an existing frozen torch::nn::Linear into a LoRALinear by
// stealing its weight tensor. The returned module is register_module()-ready.
LoRALinear make_lora_from_linear(const torch::nn::Linear& base,
                                 const LoRAConfig&         lora_cfg);
