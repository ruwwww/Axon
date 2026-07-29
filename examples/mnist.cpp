#include <iostream>
#include <iomanip>
#include <cstring>
#include "axon/data/dataloader.h"
#include "axon/data/mnist.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/linear.h"
#include "axon/nn/sgd.h"
#include "axon/runtime/runtime.h"

using namespace axon;

int main(int argc, char** argv) {
    std::string data_path = (argc > 1) ? argv[1] : ".";

    Runtime rt;

    std::cout << "Loading MNIST from: " << data_path << std::endl;
    MNIST train_dataset(rt, data_path, true);
    std::cout << "Train samples: " << train_dataset.size() << std::endl;

    DataLoader loader(train_dataset, 64, true);

    Linear model(rt, 784, 10);
    SGD optimizer(rt, model.parameters(), 0.01f);

    int epochs = 3;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int batches = 0;

        auto batches_data = loader.iter();
        for (auto& batch : batches_data) {
            optimizer.zero_grad();

            auto& inputs = batch.inputs;
            auto& targets = batch.targets;

            auto output = model.forward(rt, inputs);
            if (!output) {
                std::cerr << "Forward failed: " << output.error().message << std::endl;
                return 1;
            }

            auto loss = CrossEntropyLossOp::forward(rt, output.value(), targets);
            if (!loss) {
                std::cerr << "Loss forward failed: " << loss.error().message << std::endl;
                return 1;
            }

            total_loss += loss.value().data<float>()[0];

            auto backward = rt.autograd().backward(rt, loss.value());
            if (!backward) {
                std::cerr << "Backward failed: " << backward.error().message << std::endl;
                return 1;
            }

            // Copy gradients from autograd to parameter grads
            auto& grads = rt.autograd().gradients();
            for (auto* param : model.parameters()) {
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
            if (!step_result) {
                std::cerr << "Step failed: " << step_result.error().message << std::endl;
                return 1;
            }

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
