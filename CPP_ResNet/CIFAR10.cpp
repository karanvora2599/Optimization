// CIFAR10.cpp
#include "CIFAR10.h"
#include <fstream>
#include <iostream>

CIFAR10::CIFAR10(const std::string& root, bool train) {
    load_data(root, train);
}

void CIFAR10::load_data(const std::string& root, bool train) {
    std::string base_folder = root + "/cifar-10-batches-bin/";
    std::vector<std::string> files;
    if (train) {
        for (int i = 1; i <= 5; ++i) {
            files.push_back(base_folder + "data_batch_" + std::to_string(i) + ".bin");
        }
    } else {
        files.push_back(base_folder + "test_batch.bin");
    }

    for (const auto& file : files) {
        std::ifstream in(file, std::ios::binary);
        if (in.is_open()) {
            const int64_t num_images = 10000;
            const int64_t image_size = 3 * 32 * 32;
            const int64_t record_size = 1 + image_size;  // label + image

            std::vector<char> buffer(record_size * num_images);
            in.read(buffer.data(), buffer.size());

            for (int64_t i = 0; i < num_images; ++i) {
                int64_t label = static_cast<unsigned char>(buffer[i * record_size]);
                auto data_ptr = buffer.data() + i * record_size + 1;
                std::vector<unsigned char> image_data(data_ptr, data_ptr + image_size);

                torch::Tensor image = torch::from_blob(image_data.data(), {3, 32, 32}, torch::kUInt8).clone().to(torch::kFloat32).div(255);
                images_.push_back(image);
                targets_.push_back(torch::tensor(label, torch::kInt64));
            }
        } else {
            std::cerr << "Error opening file: " << file << std::endl;
        }
    }
}

torch::data::Example<> CIFAR10::get(size_t index) {
    return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR10::size() const {
    return images_.size();
}