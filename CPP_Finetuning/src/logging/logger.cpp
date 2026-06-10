#include "logging/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ANSI color codes (work in Windows Terminal / VS Code terminal)
namespace ansi {
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* BOLD    = "\033[1m";
    constexpr const char* GREEN   = "\033[32m";
    constexpr const char* CYAN    = "\033[36m";
    constexpr const char* YELLOW  = "\033[33m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* RED     = "\033[31m";
    constexpr const char* GRAY    = "\033[90m";
    constexpr const char* CLEAR_LINE = "\033[2K\r";
    constexpr const char* CURSOR_UP  = "\033[1A";
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────
Logger::Logger(const std::string& log_dir, int64_t max_steps, int64_t log_every)
    : log_dir_(log_dir), max_steps_(max_steps), log_every_(log_every)
{
    fs::create_directories(log_dir);

    // Open CSV
    std::string csv_path = log_dir + "/train_log.csv";
    csv_.open(csv_path);
    csv_ << "step,loss,perplexity,lr,tokens_per_sec,grad_norm,gpu_mem_mb,"
            "elapsed_sec,eta_sec\n";
    std::cout << ansi::GRAY << "[Logger] CSV log → " << csv_path
              << ansi::RESET << "\n";
}

Logger::~Logger() {
    if (csv_.is_open()) csv_.close();
}

// ─── on_train_start ───────────────────────────────────────────────────────────
void Logger::on_train_start(bool cuda, int64_t trainable_params, int64_t total_tokens) {
    start_time_ = std::chrono::steady_clock::now();
    started_ = true;
    std::cout << "\n";
    std::cout << ansi::BOLD << ansi::CYAN
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║          GPT-2 LoRA Fine-Tuning — C++/LibTorch      ║\n"
              << "╚══════════════════════════════════════════════════════╝\n"
              << ansi::RESET;
    std::cout << ansi::BOLD << "  Device       : " << ansi::RESET
              << (cuda ? ansi::GREEN : ansi::YELLOW)
              << (cuda ? "CUDA (GPU)" : "CPU") << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  LoRA params  : " << ansi::RESET
              << ansi::CYAN << trainable_params << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Dataset size : " << ansi::RESET
              << ansi::CYAN << total_tokens << " tokens" << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Max steps    : " << ansi::RESET
              << ansi::CYAN << max_steps_ << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Log every    : " << ansi::RESET
              << ansi::CYAN << log_every_ << " steps" << ansi::RESET << "\n";
    std::cout << "\n";
}

// ─── log_step ─────────────────────────────────────────────────────────────────
void Logger::log_step(const StepStats& s) {
    render_bar(s);
    write_csv_row(s);
}

// ─── Progress bar + stats block ───────────────────────────────────────────────
void Logger::render_bar(const StepStats& s) {
    // Erase previous block
    for (int i = 0; i < last_bar_lines_; ++i)
        std::cout << ansi::CURSOR_UP << ansi::CLEAR_LINE;

    // Progress bar
    constexpr int BAR_W = 40;
    float pct = static_cast<float>(s.step) / static_cast<float>(s.max_steps);
    int filled = static_cast<int>(pct * BAR_W);

    std::ostringstream bar;
    bar << ansi::BOLD << ansi::GREEN << "[";
    for (int i = 0; i < filled;    ++i) bar << "█";
    for (int i = filled; i < BAR_W; ++i) bar << "░";
    bar << "] " << ansi::RESET;
    bar << ansi::BOLD << std::setw(5) << s.step << "/" << s.max_steps
        << " (" << std::fixed << std::setprecision(1) << (pct * 100.0f) << "%)"
        << ansi::RESET;

    // Loss / perplexity row
    std::ostringstream stats1;
    // Color loss: green < 2, yellow < 4, red otherwise
    const char* loss_col = (s.loss < 2.0f) ? ansi::GREEN
                         : (s.loss < 4.0f) ? ansi::YELLOW : ansi::RED;
    stats1 << "  " << ansi::BOLD << "loss" << ansi::RESET << " "
           << loss_col << std::fixed << std::setprecision(4) << s.loss << ansi::RESET
           << "  " << ansi::BOLD << "ppl" << ansi::RESET << " "
           << ansi::CYAN << std::fixed << std::setprecision(2) << s.perplexity << ansi::RESET
           << "  " << ansi::BOLD << "lr" << ansi::RESET << " "
           << ansi::MAGENTA << std::scientific << std::setprecision(2) << s.lr << ansi::RESET;

    // Speed / memory row
    std::ostringstream stats2;
    stats2 << "  " << ansi::BOLD << "tok/s" << ansi::RESET << " "
           << ansi::CYAN << std::fixed << std::setprecision(0) << s.tokens_per_sec << ansi::RESET;
    if (s.gpu_mem_mb > 0.0f) {
        stats2 << "  " << ansi::BOLD << "GPU" << ansi::RESET << " "
               << ansi::YELLOW << std::fixed << std::setprecision(0)
               << s.gpu_mem_mb << " MB" << ansi::RESET;
    }
    stats2 << "  " << ansi::BOLD << "grad" << ansi::RESET << " "
           << ansi::GRAY << std::fixed << std::setprecision(3) << s.grad_norm << ansi::RESET;

    // Time row
    std::ostringstream stats3;
    stats3 << "  " << ansi::BOLD << "elapsed" << ansi::RESET << " "
           << ansi::GRAY << format_duration(s.elapsed_sec) << ansi::RESET
           << "  " << ansi::BOLD << "ETA" << ansi::RESET << " "
           << ansi::GRAY << format_duration(s.eta_sec) << ansi::RESET;

    std::cout << bar.str()     << "\n"
              << stats1.str()  << "\n"
              << stats2.str()  << "\n"
              << stats3.str()  << "\n";
    std::cout.flush();
    last_bar_lines_ = 4;
}

// ─── CSV row ──────────────────────────────────────────────────────────────────
void Logger::write_csv_row(const StepStats& s) {
    if (!csv_.is_open()) return;
    csv_ << s.step << ","
         << std::fixed << std::setprecision(6)
         << s.loss << ","
         << s.perplexity << ","
         << std::scientific << std::setprecision(6) << s.lr << ","
         << std::fixed << std::setprecision(1) << s.tokens_per_sec << ","
         << std::setprecision(4) << s.grad_norm << ","
         << std::setprecision(1) << s.gpu_mem_mb << ","
         << std::setprecision(2) << s.elapsed_sec << ","
         << s.eta_sec << "\n";
    csv_.flush();
}

// ─── on_checkpoint ───────────────────────────────────────────────────────────
void Logger::on_checkpoint(int64_t step, const std::string& path) {
    // Print below the progress block
    std::cout << ansi::BOLD << ansi::GREEN
              << "  ✓ Checkpoint saved [step " << step << "]: "
              << ansi::RESET << ansi::GRAY << path << ansi::RESET << "\n";
    // Reset bar so next log_step redraws cleanly from here
    last_bar_lines_ = 0;
}

// ─── on_train_end ─────────────────────────────────────────────────────────────
void Logger::on_train_end(float best_loss) {
    // Reset cursor tracking so final print is clean
    last_bar_lines_ = 0;
    double total_sec = elapsed();

    std::cout << "\n";
    std::cout << ansi::BOLD << ansi::CYAN
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║                   Training Complete                 ║\n"
              << "╚══════════════════════════════════════════════════════╝\n"
              << ansi::RESET;
    std::cout << ansi::BOLD << "  Best loss    : " << ansi::RESET
              << ansi::GREEN << std::fixed << std::setprecision(4) << best_loss << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Best ppl     : " << ansi::RESET
              << ansi::CYAN << std::fixed << std::setprecision(2)
              << std::exp(best_loss) << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Total time   : " << ansi::RESET
              << ansi::GRAY << format_duration(total_sec) << ansi::RESET << "\n";
    std::cout << ansi::BOLD << "  Log CSV      : " << ansi::RESET
              << ansi::GRAY << log_dir_ + "/train_log.csv" << ansi::RESET << "\n";
    std::cout << "\n";

    // Write JSON summary
    std::string json_path = log_dir_ + "/summary.json";
    std::ofstream jf(json_path);
    if (jf.is_open()) {
        jf << "{\n"
           << "  \"best_loss\": "        << std::fixed << std::setprecision(6) << best_loss << ",\n"
           << "  \"best_perplexity\": "  << std::fixed << std::setprecision(4) << std::exp(best_loss) << ",\n"
           << "  \"total_steps\": "      << max_steps_ << ",\n"
           << "  \"training_time_sec\": "<< std::fixed << std::setprecision(2) << total_sec << "\n"
           << "}\n";
        std::cout << ansi::BOLD << "  Summary JSON : " << ansi::RESET
                  << ansi::GRAY << json_path << ansi::RESET << "\n";
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
double Logger::elapsed() const {
    if (!started_) return 0.0;
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

std::string Logger::format_duration(double sec) const {
    if (sec < 0) sec = 0;
    int h  = static_cast<int>(sec) / 3600;
    int m  = (static_cast<int>(sec) % 3600) / 60;
    int s  = static_cast<int>(sec) % 60;
    std::ostringstream oss;
    if (h > 0)
        oss << h << "h" << std::setw(2) << std::setfill('0') << m << "m";
    else if (m > 0)
        oss << m << "m" << std::setw(2) << std::setfill('0') << s << "s";
    else
        oss << s << "s";
    return oss.str();
}
