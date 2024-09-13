// CIFAR10.h
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
private:
    std::vector<torch::Tensor> images_, targets_;
public:
    explicit CIFAR10(const std::string& root, bool train = true);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    // Function to load data
    void load_data(const std::string& root, bool train);
};