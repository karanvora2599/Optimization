#include "training/trainer.h"
#include "model/gpt2.h"
#include "data/dataset.h"
#include "logging/logger.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;
using clock_t_ = std::chrono::steady_clock;

// ─── Constructor ──────────────────────────────────────────────────────────────
Trainer::Trainer(GPT2Model model, const TrainerConfig& cfg)
    : model_(std::move(model)),
      cfg_(cfg),
      device_(cfg.use_cuda && torch::cuda::is_available()
                  ? torch::kCUDA : torch::kCPU),
      optimizer_(
          model_->lora_parameters(),
          torch::optim::AdamWOptions(cfg.lr)
              .betas({cfg.beta1, cfg.beta2})
              .weight_decay(cfg.weight_decay)),
      logger_(cfg.log_dir, cfg.max_steps, cfg.log_every)
{
    model_->to(device_);
    fs::create_directories(cfg_.checkpoint_dir);
}

// ─── LR schedule: linear warmup + cosine decay ────────────────────────────────
float Trainer::compute_lr(int64_t step) const {
    float min_lr = cfg_.lr * 0.1f;
    if (step < cfg_.warmup_steps)
        return cfg_.lr * static_cast<float>(step + 1) / cfg_.warmup_steps;

    float decay_steps = static_cast<float>(cfg_.max_steps - cfg_.warmup_steps);
    float t = static_cast<float>(step - cfg_.warmup_steps) / decay_steps;
    float cosinef = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t));
    return min_lr + (cfg_.lr - min_lr) * cosinef;
}

// ─── Grad norm ────────────────────────────────────────────────────────────────
float Trainer::compute_grad_norm() const {
    float total = 0.0f;
    for (const auto& p : model_->lora_parameters()) {
        if (p.grad().defined())
            total += p.grad().norm().item<float>() *
                     p.grad().norm().item<float>();
    }
    return std::sqrt(total);
}

// ─── GPU memory (MB) ─────────────────────────────────────────────────────────
static float gpu_mem_mb() {
#ifdef CUDA_VERSION
    if (!torch::cuda::is_available()) return 0.0f;
    // Returns bytes allocated on device 0
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return static_cast<float>(total_b - free_b) / (1024.0f * 1024.0f);
#else
    if (!torch::cuda::is_available()) return 0.0f;
    // Fallback: use LibTorch's reserved memory API
    return static_cast<float>(
               torch::cuda::memory_reserved(0)) / (1024.0f * 1024.0f);
#endif
}

// ─── Checkpoint helpers ───────────────────────────────────────────────────────
void Trainer::save_lora(int64_t step) const {
    std::string path = cfg_.checkpoint_dir + "/lora_step" +
                       std::to_string(step) + ".pt";
    auto params = model_->lora_parameters();
    torch::save(params, path);
    logger_.on_checkpoint(step, path);
}

void Trainer::load_lora(const std::string& path) {
    std::vector<torch::Tensor> loaded;
    torch::load(loaded, path);
    auto current = model_->lora_parameters();
    TORCH_CHECK(loaded.size() == current.size(),
                "Checkpoint param count mismatch");
    torch::NoGradGuard ng;
    for (size_t i = 0; i < current.size(); ++i)
        current[i].copy_(loaded[i]);
    std::cout << "[Trainer] LoRA weights loaded from: " << path << "\n";
}

// ─── Training loop ────────────────────────────────────────────────────────────
void Trainer::train(TextDataset& dataset) {
    model_->train();

    const int64_t n_samples = static_cast<int64_t>(dataset.size());
    TORCH_CHECK(n_samples > 0, "Dataset is empty");

    // Count trainable params and total tokens for the startup banner
    int64_t trainable_params = 0;
    for (const auto& p : model_->lora_parameters())
        trainable_params += p.numel();
    int64_t total_tokens = n_samples * cfg_.block_size;

    logger_.on_train_start(device_.is_cuda(), trainable_params, total_tokens);

    // Shuffled index pool
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float   running_loss    = 0.0f;
    float   best_loss       = 1e9f;
    int64_t accum_count     = 0;
    int64_t log_count       = 0;
    float   running_gn      = 0.0f;

    // For tokens-per-second measurement
    auto   step_start = clock_t_::now();
    int64_t tokens_since_log = 0;

    optimizer_.zero_grad();

    for (int64_t step = 0; step < cfg_.max_steps; ++step) {
        // Refresh shuffle each epoch
        if ((step * cfg_.batch_size) % n_samples == 0)
            std::shuffle(indices.begin(), indices.end(), rng);

        // Build mini-batch
        size_t start = (step * cfg_.batch_size) % n_samples;
        std::vector<size_t> batch_idx(
            indices.begin() + start,
            indices.begin() + std::min(start + (size_t)cfg_.batch_size,
                                       (size_t)n_samples));

        auto [input_ids, labels] = dataset.make_batch(batch_idx);
        input_ids = input_ids.to(device_);
        labels    = labels.to(device_);
        tokens_since_log += (int64_t)batch_idx.size() * cfg_.block_size;

        // Update learning rate
        float current_lr = compute_lr(step);
        for (auto& pg : optimizer_.param_groups())
            pg.options().set_lr(current_lr);

        // Forward + loss
        torch::Tensor loss;
        {
            auto logits = model_->forward(input_ids);  // [B, T, vocab]
            int64_t B = logits.size(0);
            int64_t T = logits.size(1);
            int64_t V = logits.size(2);

            loss = torch::nn::functional::cross_entropy(
                logits.view({B * T, V}),
                labels.view({B * T}));

            (loss / cfg_.grad_accum).backward();
        }

        float step_loss = loss.item<float>();
        running_loss += step_loss;
        ++accum_count;
        ++log_count;

        // Optimizer step after accumulation
        if (accum_count == cfg_.grad_accum) {
            running_gn += compute_grad_norm();
            torch::nn::utils::clip_grad_norm_(
                model_->lora_parameters(), cfg_.grad_clip);
            optimizer_.step();
            optimizer_.zero_grad();
            accum_count = 0;
        }

        // ── Log every N steps ────────────────────────────────────────────────
        if ((step + 1) % cfg_.log_every == 0) {
            auto now = clock_t_::now();
            double dt = std::chrono::duration<double>(now - step_start).count();
            step_start = now;

            float avg_loss = running_loss / static_cast<float>(log_count);
            float avg_gn   = running_gn   / std::max(1.0f,
                                 static_cast<float>(log_count / cfg_.grad_accum));
            running_loss = 0.0f;
            running_gn   = 0.0f;
            log_count    = 0;

            float tok_s = static_cast<float>(tokens_since_log) /
                          static_cast<float>(dt > 0 ? dt : 1e-9);
            tokens_since_log = 0;

            double elapsed = logger_.elapsed();
            double eta     = 0.0;
            if (step + 1 > 0)
                eta = elapsed / (step + 1) * (cfg_.max_steps - step - 1);

            best_loss = std::min(best_loss, avg_loss);

            StepStats s;
            s.step         = step + 1;
            s.max_steps    = cfg_.max_steps;
            s.loss         = avg_loss;
            s.perplexity   = std::exp(avg_loss);
            s.lr           = current_lr;
            s.tokens_per_sec = tok_s;
            s.grad_norm    = avg_gn;
            s.gpu_mem_mb   = gpu_mem_mb();
            s.elapsed_sec  = elapsed;
            s.eta_sec      = eta;

            logger_.log_step(s);
        }

        // Checkpoint
        if ((step + 1) % cfg_.save_every == 0)
            save_lora(step + 1);
    }

    // Final checkpoint
    save_lora(cfg_.max_steps);
    logger_.on_train_end(best_loss);
}
