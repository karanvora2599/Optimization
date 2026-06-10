#pragma once
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>

// ─── TrainStep — one row of statistics ────────────────────────────────────────
struct StepStats {
    int64_t step;
    int64_t max_steps;
    float   loss;
    float   perplexity;   // exp(loss)
    float   lr;
    float   tokens_per_sec;
    float   grad_norm;
    float   gpu_mem_mb;   // 0 if CPU
    double  elapsed_sec;
    double  eta_sec;
};

// ─── Logger ───────────────────────────────────────────────────────────────────
// Outputs:
//   1. Colored, in-place progress bar to stdout
//   2. CSV log file (one row per logged step)
//   3. JSON summary written at end of training
class Logger {
public:
    // log_dir: directory where train_log.csv and summary.json are written
    explicit Logger(const std::string& log_dir, int64_t max_steps, int64_t log_every);
    ~Logger();

    // Call at the START of training (prints header + opens files)
    void on_train_start(bool cuda, int64_t trainable_params, int64_t total_tokens);

    // Call every step that should be logged — prints progress bar + writes CSV row
    void log_step(const StepStats& s);

    // Call when a checkpoint is saved
    void on_checkpoint(int64_t step, const std::string& path);

    // Call at end of training — prints summary + writes summary.json
    void on_train_end(float best_loss);

    // Returns seconds since on_train_start()
    double elapsed() const;

private:
    void        render_bar(const StepStats& s);
    void        write_csv_row(const StepStats& s);
    std::string format_duration(double sec) const;
    std::string log_dir_;
    int64_t     max_steps_;
    int64_t     log_every_;
    std::ofstream csv_;
    std::chrono::steady_clock::time_point start_time_;
    bool        started_{false};
    int         last_bar_lines_{0};
};
