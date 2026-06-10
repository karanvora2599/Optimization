#include "model/gpt2.h"
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

// ═════════════════════════════════════════════════════════════════════════════
// CausalSelfAttention
// ═════════════════════════════════════════════════════════════════════════════
CausalSelfAttentionImpl::CausalSelfAttentionImpl(const GPT2Config& cfg)
    : n_head(cfg.n_head), n_embd(cfg.n_embd)
{
    TORCH_CHECK(cfg.n_embd % cfg.n_head == 0,
                "n_embd must be divisible by n_head");

    // Combined Q, K, V projection (3 x n_embd output)
    c_attn   = register_module("c_attn",
                    torch::nn::Linear(cfg.n_embd, 3 * cfg.n_embd));
    c_proj   = register_module("c_proj",
                    torch::nn::Linear(cfg.n_embd, cfg.n_embd));
    attn_drop  = register_module("attn_drop",
                    torch::nn::Dropout(cfg.dropout));
    resid_drop = register_module("resid_drop",
                    torch::nn::Dropout(cfg.dropout));

    // Causal mask — registered as a buffer so it moves with .to(device)
    auto mask = torch::ones({1, 1, cfg.block_size, cfg.block_size})
                    .tril()
                    .view({1, 1, cfg.block_size, cfg.block_size});
    register_buffer("bias", mask);
    bias = mask;
}

torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
    auto sizes  = x.sizes();
    int64_t B   = sizes[0];
    int64_t T   = sizes[1];
    int64_t C   = sizes[2];   // == n_embd
    int64_t hs  = C / n_head; // head size

    // Project to Q, K, V
    auto qkv = c_attn(x);  // [B, T, 3*C]
    auto chunks = qkv.split(n_embd, /*dim=*/2);
    auto q = chunks[0], k = chunks[1], v = chunks[2];

    // Reshape to [B, n_head, T, hs]
    auto reshape = [&](torch::Tensor t) {
        return t.view({B, T, n_head, hs}).transpose(1, 2).contiguous();
    };
    q = reshape(q);
    k = reshape(k);
    v = reshape(v);

    // Scaled dot-product attention with causal mask
    auto scale = 1.0 / std::sqrt(static_cast<double>(hs));
    auto att   = torch::matmul(q, k.transpose(-2, -1)) * scale; // [B, nh, T, T]
    att = att.masked_fill(
        bias.slice(/*dim=*/2, 0, T).slice(/*dim=*/3, 0, T) == 0,
        -1e10f);
    att = torch::softmax(att, /*dim=*/-1);
    att = attn_drop(att);

    // Weighted sum over values
    auto y = torch::matmul(att, v);  // [B, nh, T, hs]
    y = y.transpose(1, 2).contiguous().view({B, T, C});

    return resid_drop(c_proj(y));
}

// ═════════════════════════════════════════════════════════════════════════════
// Mlp
// ═════════════════════════════════════════════════════════════════════════════
MlpImpl::MlpImpl(const GPT2Config& cfg) {
    c_fc   = register_module("c_fc",
                 torch::nn::Linear(cfg.n_embd, 4 * cfg.n_embd));
    c_proj = register_module("c_proj",
                 torch::nn::Linear(4 * cfg.n_embd, cfg.n_embd));
    act    = register_module("act", torch::nn::GELU());
    drop   = register_module("drop", torch::nn::Dropout(cfg.dropout));
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
    return drop(c_proj(act(c_fc(x))));
}

// ═════════════════════════════════════════════════════════════════════════════
// Block
// ═════════════════════════════════════════════════════════════════════════════
BlockImpl::BlockImpl(const GPT2Config& cfg) {
    ln_1 = register_module("ln_1",
               torch::nn::LayerNorm(
                   torch::nn::LayerNormOptions({cfg.n_embd})));
    attn = register_module("attn", CausalSelfAttention(cfg));
    ln_2 = register_module("ln_2",
               torch::nn::LayerNorm(
                   torch::nn::LayerNormOptions({cfg.n_embd})));
    mlp  = register_module("mlp", Mlp(cfg));
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    x = x + attn(ln_1(x));  // pre-norm + residual
    x = x + mlp(ln_2(x));
    return x;
}

// ═════════════════════════════════════════════════════════════════════════════
// GPT2Model
// ═════════════════════════════════════════════════════════════════════════════
GPT2ModelImpl::GPT2ModelImpl(const GPT2Config& cfg) : config(cfg) {
    wte     = register_module("wte",
                  torch::nn::Embedding(cfg.vocab_size, cfg.n_embd));
    wpe     = register_module("wpe",
                  torch::nn::Embedding(cfg.block_size, cfg.n_embd));
    drop    = register_module("drop", torch::nn::Dropout(cfg.dropout));
    h       = register_module("h",   torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layer; ++i)
        h->push_back(Block(cfg));
    ln_f    = register_module("ln_f",
                  torch::nn::LayerNorm(
                      torch::nn::LayerNormOptions({cfg.n_embd})));
    // Weight tying: lm_head shares weight with wte
    lm_head = register_module("lm_head",
                  torch::nn::Linear(
                      torch::nn::LinearOptions(cfg.n_embd, cfg.vocab_size)
                          .bias(false)));
    lm_head->weight = wte->weight;  // tied
}

torch::Tensor GPT2ModelImpl::forward(torch::Tensor input_ids) {
    int64_t T = input_ids.size(1);
    TORCH_CHECK(T <= config.block_size,
                "Sequence length ", T, " exceeds block_size ", config.block_size);

    auto pos = torch::arange(0, T, torch::TensorOptions()
                                       .dtype(torch::kLong)
                                       .device(input_ids.device()));

    auto tok_emb = wte(input_ids);          // [B, T, n_embd]
    auto pos_emb = wpe(pos.unsqueeze(0));   // [1, T, n_embd]
    auto x = drop(tok_emb + pos_emb);

    for (auto& block_module : *h)
        x = block_module->as<BlockImpl>()->forward(x);

    x = ln_f(x);
    return lm_head(x);  // [B, T, vocab_size]
}

void GPT2ModelImpl::load_pretrained(const std::string& weight_path) {
    std::cout << "[GPT2] Loading pretrained weights from: " << weight_path << "\n";
    auto jit_module = torch::jit::load(weight_path, torch::kCPU);

    // Build a flat map of name -> tensor from the JIT module
    std::unordered_map<std::string, torch::Tensor> state;
    for (const auto& p : jit_module.named_parameters()) {
        // Reverse the double-underscore substitution done in export_gpt2.py
        std::string name = p.name;
        std::string result;
        for (size_t i = 0; i < name.size(); ) {
            if (i + 1 < name.size() && name[i] == '_' && name[i+1] == '_') {
                result += '.';
                i += 2;
            } else {
                result += name[i++];
            }
        }
        state[result] = p.value;
    }

    // Copy into this module's parameters by name
    for (auto& p : this->named_parameters(/*recurse=*/true)) {
        auto it = state.find(p.key());
        if (it == state.end()) {
            // HuggingFace uses "transformer." prefix
            it = state.find("transformer." + p.key());
        }
        if (it != state.end()) {
            torch::NoGradGuard no_grad;
            p.value().copy_(it->second);
        } else {
            std::cerr << "[GPT2] WARNING: no pretrained weight for: "
                      << p.key() << "\n";
        }
    }
    std::cout << "[GPT2] Weights loaded successfully.\n";
}

void GPT2ModelImpl::freeze_base_weights() {
    for (auto& p : this->named_parameters(/*recurse=*/true)) {
        // LoRA parameters are named lora_A / lora_B — keep those trainable
        if (p.key().find("lora_") == std::string::npos)
            p.value().set_requires_grad(false);
    }
}

std::vector<torch::Tensor> GPT2ModelImpl::lora_parameters() {
    std::vector<torch::Tensor> params;
    for (auto& p : this->named_parameters(/*recurse=*/true)) {
        if (p.key().find("lora_") != std::string::npos)
            params.push_back(p.value());
    }
    return params;
}
