// main.cpp
#include <torch/torch.h>
#include "resnet.h"
#include "CIFAR10.h"
#include <iostream>
#include <chrono>
#include <iomanip>  // For setting output precision
#include <limits>   // For setting initial best validation loss
#include <cmath>    // For std::ceil

int main() {
    try {
        // Check for CUDA availability
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::kCUDA;
        } else {
            std::cout << "CUDA not available. Using CPU." << std::endl;
        }

        // Specify the ResNet layers (e.g., ResNet-18 = {2, 2, 2, 2})
        std::vector<int64_t> layers = {2, 2, 2, 2};

        // Instantiate the model for CIFAR-10 with the specified number of layers
        std::cout << "Instantiating model..." << std::endl;
        ResNet model(layers, 10);  // Pass the vector explicitly

        model->to(device);

        // Load CIFAR-10 dataset
        std::string dataset_path = "C:/Users/karan/Documents/Optimization Techniques/CPP_ResNet/data";
        auto train_dataset = CIFAR10(dataset_path, /*train=*/true)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                      {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        auto test_dataset = CIFAR10(dataset_path, /*train=*/false)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                      {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        // Get dataset sizes
        const int64_t train_dataset_size = static_cast<int64_t>(train_dataset.size().value());
        const int64_t test_dataset_size = static_cast<int64_t>(test_dataset.size().value());

        // Create data loaders with more workers for asynchronous loading
        const int64_t batch_size = 64;
        const int num_workers = 4;  // Increased number of workers

        auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(batch_size)
                .workers(num_workers)
                .enforce_ordering(false));

        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(batch_size)
                .workers(num_workers));

        // Set up optimizer and loss function
        torch::optim::SGD optimizer(
            model->parameters(),
            torch::optim::SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(5e-4));

        // Use StepLR scheduler
        torch::optim::StepLR scheduler(optimizer, /*step_size=*/5, /*gamma=*/0.1);

        const int64_t num_epochs = 30;

        // Variables to track the best validation loss for checkpointing
        double best_valid_loss = std::numeric_limits<double>::infinity();

        // Start total training time
        auto training_start = std::chrono::steady_clock::now();

        for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
            auto epoch_start = std::chrono::steady_clock::now();
            std::cout << "Epoch: " << epoch << "/" << num_epochs << std::endl;

            // Start tracking training time
            auto train_start = std::chrono::steady_clock::now();

            model->train();
            double running_loss = 0.0;
            int64_t epoch_correct = 0;
            int64_t total_samples = 0;
            int64_t batch_idx = 0;

            size_t total_batches = std::ceil(static_cast<double>(train_dataset_size) / batch_size);
            auto batch_start_time = std::chrono::steady_clock::now();

            for (torch::data::Example<>& batch : *train_loader) {
                batch_idx++;

                // Timing for batch processing
                auto batch_end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> batch_duration = batch_end_time - batch_start_time;
                double batch_time = batch_duration.count();
                double iter_per_sec = 1.0 / batch_time;
                batch_start_time = std::chrono::steady_clock::now();

                // Move data to device
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                // Forward pass
                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);

                // Backward pass and optimization
                loss.backward();
                optimizer.step();

                // Update running metrics
                running_loss += loss.item<double>();
                auto pred = output.argmax(1);
                epoch_correct += pred.eq(target).sum().item<int64_t>();
                total_samples += data.size(0);

                // Print progress every 10 batches
                if (batch_idx % 10 == 0 || batch_idx == total_batches) {
                    double progress = static_cast<double>(batch_idx) / total_batches;
                    int bar_width = 50;
                    int pos = static_cast<int>(bar_width * progress);
                    std::cout << "[";
                    for (int i = 0; i < bar_width; ++i) {
                        if (i < pos) std::cout << "=";
                        else if (i == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1)
                              << (progress * 100.0) << "% "
                              << "Loss: " << std::setprecision(4) << loss.item<double>() << " "
                              << "Iter/sec: " << std::setprecision(2) << iter_per_sec << "     \r";
                    std::cout.flush();
                }
            }
            std::cout << std::endl;  // Move to next line after the loop

            // End tracking training time
            auto train_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> train_duration = train_end - train_start;
            std::cout << "Train time: " << train_duration.count() << " seconds" << std::endl;

            double train_loss = running_loss / batch_idx;
            double train_accuracy = static_cast<double>(epoch_correct) / total_samples * 100.0;

            // Start tracking evaluation time
            auto eval_start = std::chrono::steady_clock::now();

            // Evaluate on test data
            model->eval();
            double val_running_loss = 0.0;
            int64_t val_correct = 0;
            int64_t val_total_samples = 0;
            int64_t val_batch_idx = 0;

            {
                torch::NoGradGuard no_grad;
                for (torch::data::Example<>& batch : *test_loader) {
                    val_batch_idx++;
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    // Forward pass
                    auto output = model->forward(data);
                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    // Update running metrics
                    val_running_loss += loss.item<double>();
                    auto pred = output.argmax(1);
                    val_correct += pred.eq(target).sum().item<int64_t>();
                    val_total_samples += data.size(0);
                }
            }

            double val_loss = val_running_loss / val_batch_idx;
            double val_accuracy = static_cast<double>(val_correct) / val_total_samples * 100.0;

            // End tracking evaluation time
            auto eval_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "Evaluation time: " << eval_duration.count() << " seconds" << std::endl;

            // Check if validation loss improved and save model
            // if (val_loss < best_valid_loss) {
            //     best_valid_loss = val_loss;
            //     torch::save(model, "best_model.pt");
            // }

            auto epoch_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

            // Print epoch statistics
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "Train Accuracy: " << train_accuracy << "%, Train Loss: " << train_loss << std::endl;
            std::cout << "Test Accuracy: " << val_accuracy << "%, Test Loss: " << val_loss << std::endl;
            std::cout << "Epoch time: " << epoch_duration.count() << " seconds" << std::endl << std::endl;

            // Step the scheduler after each epoch
            scheduler.step();
        }

        // Total training time
        auto training_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> training_duration = training_end - training_start;
        std::cout << "Total training time: " << training_duration.count() << " seconds" << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "C10 Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown Exception occurred!" << std::endl;
        return -1;
    }

    return 0;
}