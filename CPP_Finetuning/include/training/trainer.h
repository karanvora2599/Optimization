#pragma once
#include <torch/torch.h>
#include <string>
#include "config.h"
#include "logging/logger.h"

struct TrainerConfig {
    // Optimization
    float   lr              = 2e-4f;
    float   weight_decay    = 0.01f;
    float   beta1           = 0.9f;
    float   beta2           = 0.95f;
    float   grad_clip       = 1.0f;

    // Training schedule
    int64_t max_steps       = 1000;
    int64_t warmup_steps    = 50;
    int64_t grad_accum      = 4;    // accumulate N mini-batches before step
    int64_t batch_size      = 4;    // sequences per mini-batch
    int64_t block_size      = 512;  // context window (tokens)

    // Logging / checkpointing
    int64_t log_every       = 10;
    int64_t save_every      = 200;
    std::string checkpoint_dir = "checkpoints";
    std::string log_dir        = "logs";      // CSV + JSON written here

    // Device
    bool use_cuda           = true;   // will fall back to CPU if unavailable
    bool use_amp            = true;   // mixed-precision (BF16)
};

// ─── Trainer ──────────────────────────────────────────────────────────────────
// Owns the model, optimizer, and the full training loop.
// The model must have LoRA weights injected and base weights frozen before
// passing it to Trainer.
class TextDataset;

struct GPT2ModelImpl;
TORCH_MODULE_IMPL(GPT2Model, GPT2ModelImpl);

class Trainer {
public:
    Trainer(GPT2Model model, const TrainerConfig& cfg);

    // Run training for cfg.max_steps steps on the provided dataset.
    void train(TextDataset& dataset);

    // Save only the LoRA adapter parameters to a file.
    void save_lora(int64_t step) const;

    // Load LoRA adapter parameters from a file.
    void load_lora(const std::string& path);

private:
    float compute_lr(int64_t step) const;
    float compute_grad_norm() const;

    GPT2Model          model_;
    TrainerConfig      cfg_;
    torch::Device      device_;
    torch::optim::AdamW optimizer_;
    Logger             logger_;      // owns the CSV / progress bar
};
