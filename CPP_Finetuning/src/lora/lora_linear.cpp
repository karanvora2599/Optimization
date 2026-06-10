#include "lora/lora_linear.h"
#include <torch/torch.h>

// ═════════════════════════════════════════════════════════════════════════════
// LoRALinear
// ═════════════════════════════════════════════════════════════════════════════
LoRALinearImpl::LoRALinearImpl(int64_t in_features,
                               int64_t out_features,
                               const LoRAConfig& lora_cfg,
                               bool    has_bias_)
    : has_bias(has_bias_), scaling(lora_cfg.alpha / static_cast<float>(lora_cfg.rank))
{
    // Base weight — frozen: stored as a plain tensor, NOT a parameter
    weight = register_buffer("weight",
                 torch::empty({out_features, in_features}));
    if (has_bias)
        base_bias = register_buffer("base_bias",
                        torch::zeros({out_features}));

    // Trainable LoRA matrices (no bias needed in adapter)
    lora_A = register_module("lora_A",
                 torch::nn::Linear(
                     torch::nn::LinearOptions(in_features, lora_cfg.rank)
                         .bias(false)));
    lora_B = register_module("lora_B",
                 torch::nn::Linear(
                     torch::nn::LinearOptions(lora_cfg.rank, out_features)
                         .bias(false)));

    // Canonical LoRA init: Gaussian for A, zeros for B (so delta = 0 at startup)
    torch::nn::init::normal_(lora_A->weight, 0.0, 0.02);
    torch::nn::init::zeros_(lora_B->weight);
}

torch::Tensor LoRALinearImpl::forward(torch::Tensor x) {
    // Base (frozen) pass: x @ W.T
    auto base_out = torch::nn::functional::linear(x, weight,
                        has_bias ? base_bias : torch::Tensor{});
    // LoRA delta: x * A -> rank space -> B -> out, scaled
    auto lora_out = lora_B(lora_A(x)) * scaling;
    return base_out + lora_out;
}

// ─── Helper ──────────────────────────────────────────────────────────────────
LoRALinear make_lora_from_linear(const torch::nn::Linear& base,
                                 const LoRAConfig&         lora_cfg) {
    int64_t out = base->weight.size(0);
    int64_t in  = base->weight.size(1);
    bool    has_bias = (base->bias.defined());

    auto lora = LoRALinear(in, out, lora_cfg, has_bias);

    // Copy the pretrained weight into the buffer (no grad)
    {
        torch::NoGradGuard ng;
        lora->weight.copy_(base->weight);
        if (has_bias)
            lora->base_bias.copy_(base->bias);
    }
    return lora;
}
