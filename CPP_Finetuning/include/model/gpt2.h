#pragma once
#include <torch/torch.h>
#include "config.h"

// ─── Causal Self-Attention ────────────────────────────────────────────────────
struct CausalSelfAttentionImpl : torch::nn::Module {
    explicit CausalSelfAttentionImpl(const GPT2Config& cfg);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear c_attn{nullptr};  // projects input → Q, K, V (3 * n_embd)
    torch::nn::Linear c_proj{nullptr};  // output projection
    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Dropout resid_drop{nullptr};
    torch::Tensor     bias;             // causal (lower-triangular) mask [1,1,T,T]

    int64_t n_head;
    int64_t n_embd;
};
TORCH_MODULE(CausalSelfAttention);

// ─── Feed-Forward MLP ─────────────────────────────────────────────────────────
struct MlpImpl : torch::nn::Module {
    explicit MlpImpl(const GPT2Config& cfg);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear c_fc{nullptr};   // n_embd → 4 * n_embd
    torch::nn::Linear c_proj{nullptr}; // 4 * n_embd → n_embd
    torch::nn::GELU   act;
    torch::nn::Dropout drop{nullptr};
};
TORCH_MODULE(Mlp);

// ─── Transformer Block ────────────────────────────────────────────────────────
struct BlockImpl : torch::nn::Module {
    explicit BlockImpl(const GPT2Config& cfg);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::LayerNorm    ln_1{nullptr};
    CausalSelfAttention     attn{nullptr};
    torch::nn::LayerNorm    ln_2{nullptr};
    Mlp                     mlp{nullptr};
};
TORCH_MODULE(Block);

// ─── GPT-2 Language Model ─────────────────────────────────────────────────────
struct GPT2ModelImpl : torch::nn::Module {
    explicit GPT2ModelImpl(const GPT2Config& cfg);

    // Forward pass: input_ids [B, T] → logits [B, T, vocab_size]
    torch::Tensor forward(torch::Tensor input_ids);

    // Load pretrained weights exported by scripts/export_gpt2.py
    void load_pretrained(const std::string& weight_path);

    // Returns only the LoRA adapter parameters (for optimizer)
    std::vector<torch::Tensor> lora_parameters();

    // Freeze all base parameters; call before inject_lora()
    void freeze_base_weights();

    torch::nn::Embedding     wte{nullptr};  // token embeddings  [vocab, n_embd]
    torch::nn::Embedding     wpe{nullptr};  // position embeddings [T, n_embd]
    torch::nn::Dropout       drop{nullptr};
    torch::nn::ModuleList    h{nullptr};    // transformer blocks
    torch::nn::LayerNorm     ln_f{nullptr}; // final layer norm
    torch::nn::Linear        lm_head{nullptr};

    GPT2Config config;
};
TORCH_MODULE(GPT2Model);
