#include <iostream>
#include <iomanip>
#include <cstring>
#include "axon/data/cifar10.h"
#include "axon/data/dataloader.h"
#include "axon/nn/conv2d.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/flatten.h"
#include "axon/nn/linear.h"
#include "axon/nn/adamw.h"
#include "axon/nn/sequential.h"
#include "axon/runtime/runtime.h"

using namespace axon;

int main(int argc, char** argv) {
    std::string data_path = (argc > 1) ? argv[1] : ".";

    Runtime rt;

    std::cout << "Loading CIFAR-10 from: " << data_path << std::endl;
    CIFAR10 train_dataset(rt, data_path, true);
    std::cout << "Train samples: " << train_dataset.size() << std::endl;

    DataLoader loader(train_dataset, 64, true);

    // Simple ConvNet: Conv2D(3,16,3,1,1) -> ReLU -> Conv2D(16,32,3,1,1) -> ReLU -> Flatten -> Linear(32*32*32, 10)
    Conv2D conv1(rt, 3, 16, 3, 1, 1);
    Conv2D conv2(rt, 16, 32, 3, 1, 1);
    Flatten flatten;
    Linear linear(rt, 32 * 32 * 32, 10);

    // Collect all parameters
    std::vector<Parameter*> all_params;
    for (auto* p : conv1.parameters()) all_params.push_back(p);
    for (auto* p : conv2.parameters()) all_params.push_back(p);
    for (auto* p : linear.parameters()) all_params.push_back(p);

    AdamW optimizer(rt, all_params, 0.001f);

    int epochs = 3;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int batches = 0;

        auto batches_data = loader.iter();
        for (auto& batch : batches_data) {
            optimizer.zero_grad();

            auto& inputs = batch.inputs;
            auto& targets = batch.targets;

            // Forward: conv1 -> relu -> conv2 -> relu -> flatten -> linear
            auto h1 = conv1.forward(rt, inputs);
            if (!h1) { std::cerr << "conv1 failed: " << h1.error().message << std::endl; return 1; }

            auto a1 = rt.relu(h1.value());
            if (!a1) { std::cerr << "relu1 failed" << std::endl; return 1; }

            auto h2 = conv2.forward(rt, a1.value());
            if (!h2) { std::cerr << "conv2 failed: " << h2.error().message << std::endl; return 1; }

            auto a2 = rt.relu(h2.value());
            if (!a2) { std::cerr << "relu2 failed" << std::endl; return 1; }

            auto flat = flatten.forward(rt, a2.value());
            if (!flat) { std::cerr << "flatten failed" << std::endl; return 1; }

            auto output = linear.forward(rt, flat.value());
            if (!output) { std::cerr << "linear failed: " << output.error().message << std::endl; return 1; }

            auto loss = CrossEntropyLossOp::forward(rt, output.value(), targets);
            if (!loss) { std::cerr << "loss failed: " << loss.error().message << std::endl; return 1; }

            total_loss += loss.value().data<float>()[0];

            auto backward = rt.autograd().backward(rt, loss.value());
            if (!backward) { std::cerr << "backward failed: " << backward.error().message << std::endl; return 1; }

            auto& grads = rt.autograd().gradients();
            for (auto* param : all_params) {
                if (!param->trainable()) continue;
                auto it = grads.find(param->tensor().id());
                if (it != grads.end()) {
                    std::memcpy(
                        param->grad().storage()->data,
                        it->second.storage()->data,
                        it->second.storage()->size_bytes
                    );
                }
            }

            auto step_result = optimizer.step();
            if (!step_result) { std::cerr << "step failed: " << step_result.error().message << std::endl; return 1; }

            rt.autograd().clear_gradients();
            rt.autograd().graph().clear();
            batches++;
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << "  Avg Loss: " << std::fixed << std::setprecision(4)
                  << (total_loss / batches) << std::endl;
    }

    std::cout << "Training complete!" << std::endl;
    return 0;
}
