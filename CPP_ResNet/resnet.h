// resnet.h
#pragma once

#include <torch/torch.h>

// BasicBlock definition
struct BasicBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential shortcut{nullptr};
    int64_t expansion = 1;

    BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride = 1) {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 3)
            .stride(stride).padding(1).bias(false));
        bn1 = torch::nn::BatchNorm2d(planes);

        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes * expansion, 3)
            .stride(1).padding(1).bias(false));
        bn2 = torch::nn::BatchNorm2d(planes * expansion);

        shortcut = torch::nn::Sequential();
        if (stride != 1 || in_planes != planes * expansion) {
            shortcut->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes * expansion, 1)
                .stride(stride).bias(false)));
            shortcut->push_back(torch::nn::BatchNorm2d(planes * expansion));
        }

        // Register modules
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("shortcut", shortcut);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));

        auto identity = x;
        if (!shortcut->is_empty()) {
            identity = shortcut->forward(x);
        }

        out += identity;
        out = torch::relu(out);

        return out;
    }
};
TORCH_MODULE(BasicBlock);

// ResNet definition
struct ResNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::Linear linear{nullptr};

    int64_t in_planes;
    int64_t expansion = 1;

    ResNetImpl(std::vector<int64_t> num_blocks, int64_t num_classes = 10) {
        in_planes = 64;

        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3)
            .stride(1).padding(1).bias(false));
        bn1 = torch::nn::BatchNorm2d(64);

        layer1 = _make_layer(64, num_blocks[0], 1);
        layer2 = _make_layer(128, num_blocks[1], 2);
        layer3 = _make_layer(256, num_blocks[2], 2);
        layer4 = _make_layer(512, num_blocks[3], 2);

        linear = torch::nn::Linear(512 * expansion, num_classes);

        // Register modules
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("linear", linear);
    }

    torch::nn::Sequential _make_layer(int64_t planes, int64_t num_blocks, int64_t stride) {
        std::vector<int64_t> strides;
        strides.push_back(stride);
        for (int64_t i = 1; i < num_blocks; ++i) {
            strides.push_back(1);
        }

        torch::nn::Sequential layers;
        for (auto stride : strides) {
            layers->push_back(BasicBlock(in_planes, planes, stride));
            in_planes = planes * expansion;
        }
        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = torch::avg_pool2d(x, 4);
        x = x.view({x.size(0), -1});
        x = linear->forward(x);

        return x;
    }
};
TORCH_MODULE(ResNet);
