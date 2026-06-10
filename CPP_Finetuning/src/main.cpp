#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "config.h"
#include "model/gpt2.h"
#include "lora/lora_linear.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "training/trainer.h"

// ─── Simple arg parser ────────────────────────────────────────────────────────
struct Args {
    std::string mode          = "train";   // train | test_forward | generate
    std::string weights       = "gpt2_weights.pt";
    std::string vocab         = "vocab.json";
    std::string merges        = "merges.txt";
    std::string data          = "data/sample.txt";
    std::string lora          = "";
    std::string prompt        = "Once upon a time";
    int64_t     steps         = 1000;
    int64_t     batch_size    = 4;
    int64_t     block_size    = 512;
    int64_t     log_every     = 10;
    int64_t     save_every    = 200;
    int64_t     max_new_tokens= 200;
    bool        no_cuda       = false;
    std::string log_dir       = "logs";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + s);
            return argv[++i];
        };
        if      (s == "--mode")       a.mode       = next();
        else if (s == "--weights")    a.weights    = next();
        else if (s == "--vocab")      a.vocab      = next();
        else if (s == "--merges")     a.merges     = next();
        else if (s == "--data")       a.data       = next();
        else if (s == "--lora")       a.lora       = next();
        else if (s == "--prompt")     a.prompt     = next();
        else if (s == "--steps")      a.steps      = std::stoll(next());
        else if (s == "--batch-size") a.batch_size = std::stoll(next());
        else if (s == "--block-size") a.block_size = std::stoll(next());
        else if (s == "--log-every")  a.log_every  = std::stoll(next());
        else if (s == "--save-every") a.save_every = std::stoll(next());
        else if (s == "--max-tokens") a.max_new_tokens = std::stoll(next());
        else if (s == "--no-cuda")    a.no_cuda    = true;
        else if (s == "--log-dir")    a.log_dir    = next();
        else { std::cerr << "[WARN] Unknown arg: " << s << "\n"; }
    }
    return a;
}

// ─── Top-k sampling for generation ────────────────────────────────────────────
static int64_t sample_topk(torch::Tensor logits, int64_t k = 50, float temp = 0.8f) {
    logits = logits / temp;
    auto [values, indices] = logits.topk(k, /*dim=*/-1);
    auto probs = torch::softmax(values, -1);
    auto sample_idx = torch::multinomial(probs, 1);
    return indices[sample_idx.item<int64_t>()].item<int64_t>();
}

// ─── Modes ────────────────────────────────────────────────────────────────────
void run_test_forward(GPT2Model& model, const torch::Device& device) {
    model->eval();
    auto dummy = torch::zeros({1, 64}, torch::TensorOptions().dtype(torch::kLong))
                     .to(device);
    auto logits = model->forward(dummy);
    std::cout << "[test_forward] Output shape: " << logits.sizes() << "\n";
    std::cout << "[test_forward] PASS — expected [1, 64, 50257]\n";
}

void run_generate(GPT2Model& model,
                  const BPETokenizer& tok,
                  const torch::Device& device,
                  const std::string& prompt,
                  int64_t max_new_tokens)
{
    model->eval();
    torch::NoGradGuard no_grad;

    auto ids = tok.encode(prompt);
    auto input = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong))
                     .unsqueeze(0).to(device);

    std::cout << "\n[Generate] Prompt: " << prompt << "\n";
    std::cout << "[Generate] ";

    for (int64_t i = 0; i < max_new_tokens; ++i) {
        auto logits = model->forward(input);          // [1, T, vocab]
        auto next_logits = logits[0].select(0, -1);   // last position [vocab]
        int64_t next_id = sample_topk(next_logits);
        ids.push_back(next_id);

        // Append
        auto next_tok = torch::tensor(
            std::vector<int64_t>{next_id},
            torch::TensorOptions().dtype(torch::kLong)).unsqueeze(0).to(device);
        input = torch::cat({input, next_tok}, 1);

        // Keep context within block_size
        if (input.size(1) > 1024)
            input = input.slice(1, input.size(1) - 1024, input.size(1));

        std::cout << tok.decode({next_id}) << std::flush;
        if (next_id == tok.eos_token_id()) break;
    }
    std::cout << "\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    torch::Device device = (!args.no_cuda && torch::cuda::is_available())
                               ? torch::Device(torch::kCUDA)
                               : torch::Device(torch::kCPU);
    std::cout << "[Main] Using device: "
              << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    // ── Tokenizer ─────────────────────────────────────────────────────────────
    BPETokenizer tokenizer;
    if (!tokenizer.load(args.vocab, args.merges)) {
        std::cerr << "[Main] Failed to load tokenizer.\n";
        return 1;
    }

    // ── Model ─────────────────────────────────────────────────────────────────
    GPT2Config model_cfg;
    model_cfg.block_size = args.block_size;
    GPT2Model model(model_cfg);
    model->load_pretrained(args.weights);

    // ── Inject LoRA into all attention c_attn projections ─────────────────────
    {
        LoRAConfig lora_cfg;  // rank=8, alpha=16 by default
        for (int64_t i = 0; i < model_cfg.n_layer; ++i) {
            auto block = model->h->ptr<BlockImpl>(i);
            // Replace c_attn with LoRALinear
            auto lora_attn = make_lora_from_linear(block->attn->c_attn, lora_cfg);
            block->attn->c_attn = torch::nn::Linear(nullptr); // unregister old
            block->attn->register_module("c_attn_lora", lora_attn);
        }
    }

    // Freeze everything except LoRA
    model->freeze_base_weights();
    model->to(device);

    // ── Load LoRA checkpoint if provided ──────────────────────────────────────
    if (!args.lora.empty()) {
        std::vector<torch::Tensor> loaded;
        torch::load(loaded, args.lora);
        auto params = model->lora_parameters();
        TORCH_CHECK(loaded.size() == params.size(), "LoRA checkpoint size mismatch");
        torch::NoGradGuard ng;
        for (size_t i = 0; i < params.size(); ++i)
            params[i].copy_(loaded[i]);
        std::cout << "[Main] LoRA weights loaded from: " << args.lora << "\n";
    }

    // ── Dispatch mode ─────────────────────────────────────────────────────────
    if (args.mode == "test_forward") {
        run_test_forward(model, device);

    } else if (args.mode == "generate") {
        run_generate(model, tokenizer, device, args.prompt, args.max_new_tokens);

    } else if (args.mode == "train") {
        TextDataset dataset(args.data, tokenizer, args.block_size);

        TrainerConfig trainer_cfg;
        trainer_cfg.max_steps    = args.steps;
        trainer_cfg.batch_size   = args.batch_size;
        trainer_cfg.block_size   = args.block_size;
        trainer_cfg.log_every    = args.log_every;
        trainer_cfg.save_every   = args.save_every;
        trainer_cfg.use_cuda     = !args.no_cuda;
        trainer_cfg.log_dir      = args.log_dir;

        Trainer trainer(model, trainer_cfg);
        trainer.train(dataset);

    } else {
        std::cerr << "[Main] Unknown mode: " << args.mode
                  << ". Use train | test_forward | generate\n";
        return 1;
    }

    return 0;
}
