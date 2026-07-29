#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include "axon/runtime/runtime.h"
#include "axon/data/dataloader.h"
#include "axon/data/mnist.h"
#include "axon/nn/linear.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/sgd.h"

using namespace axon;

int main(int argc, char** argv) {
    std::string data_path = (argc > 1) ? argv[1] : "datasets/mnist";
    int epochs = (argc > 2) ? std::stoi(argv[2]) : 10;
    int batch_size = 64;
    float lr = 0.05f;

    Runtime rt;

    // Load real MNIST dataset
    MNIST train_dataset(rt, data_path, true);
    MNIST test_dataset(rt, data_path, false);

    DataLoader train_loader(train_dataset, batch_size, true);
    DataLoader test_loader(test_dataset, batch_size, false);

    // 2-Layer MLP Architecture: 784 -> 128 -> 10
    Linear fc1(rt, 784, 128);
    Linear fc2(rt, 128, 10);

    std::vector<Parameter*> params;
    for (auto* p : fc1.parameters()) params.push_back(p);
    for (auto* p : fc2.parameters()) params.push_back(p);

    SGD optimizer(rt, params, lr);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        float train_loss = 0.0f;
        int train_batches = 0, train_correct = 0, train_total = 0;

        // 1. Train Loop
        auto batches = train_loader.iter();
        for (auto& batch : batches) {
            optimizer.zero_grad();
            batch.inputs.set_requires_grad(true);

            auto h1 = fc1.forward(rt, batch.inputs).value();
            auto a1 = rt.relu(h1).value();
            auto out = fc2.forward(rt, a1).value();
            auto loss = CrossEntropyLossOp::forward(rt, out, batch.targets).value();

            train_loss += loss.data<float>()[0];

            int64_t bs = batch.inputs.type().shape()[0];
            const float* out_ptr = out.data<float>();
            const int64_t* targ_ptr = batch.targets.data<int64_t>();
            for (int64_t i = 0; i < bs; ++i) {
                int max_idx = 0;
                float max_val = out_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (out_ptr[i * 10 + c] > max_val) {
                        max_val = out_ptr[i * 10 + c];
                        max_idx = c;
                    }
                }
                if (max_idx == targ_ptr[i]) train_correct++;
            }
            train_total += bs;

            rt.autograd().backward(rt, loss);

            auto& grads = rt.autograd().gradients();
            for (auto* param : params) {
                if (!param->trainable()) continue;
                auto it = grads.find(param->tensor().id());
                if (it != grads.end()) {
                    std::memcpy(param->grad().storage()->data, it->second.storage()->data, it->second.storage()->size_bytes);
                }
            }

            optimizer.step();
            rt.autograd().clear_gradients();
            rt.autograd().graph().clear();
            train_batches++;
        }

        // 2. Test Evaluation Loop (No loss calculation, no backprop - pure inference)
        int test_correct = 0, test_total = 0;
        auto test_batches_data = test_loader.iter();
        for (auto& batch : test_batches_data) {
            auto h1 = fc1.forward(rt, batch.inputs).value();
            auto a1 = rt.relu(h1).value();
            auto out = fc2.forward(rt, a1).value();

            int64_t bs = batch.inputs.type().shape()[0];
            const float* out_ptr = out.data<float>();
            const int64_t* targ_ptr = batch.targets.data<int64_t>();
            for (int64_t i = 0; i < bs; ++i) {
                int max_idx = 0;
                float max_val = out_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (out_ptr[i * 10 + c] > max_val) {
                        max_val = out_ptr[i * 10 + c];
                        max_idx = c;
                    }
                }
                if (max_idx == targ_ptr[i]) test_correct++;
            }
            test_total += bs;
            rt.autograd().graph().clear();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        float train_acc = (static_cast<float>(train_correct) / train_total) * 100.0f;
        float test_acc = (static_cast<float>(test_correct) / test_total) * 100.0f;

        std::cout << "Epoch [" << std::setw(2) << epoch << "/" << epochs << "]"
                  << " | Train Loss: " << std::fixed << std::setprecision(4) << (train_loss / train_batches)
                  << " | Train Acc: " << std::setprecision(2) << train_acc << "%"
                  << " | Test Acc: " << std::setprecision(2) << test_acc << "%"
                  << " | Time: " << std::setprecision(2) << elapsed << " s\n";
    }
    return 0;
}
