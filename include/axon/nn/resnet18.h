#pragma once

#include <memory>
#include "axon/nn/batchnorm.h"
#include "axon/nn/conv2d.h"
#include "axon/nn/linear.h"
#include "axon/nn/module.h"

namespace axon {

class BasicBlock : public Module {
public:
    BasicBlock(Runtime& rt, int64_t in_planes, int64_t planes, int64_t stride = 1);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    Conv2D conv1_;
    BatchNorm bn1_;
    Conv2D conv2_;
    BatchNorm bn2_;
    std::unique_ptr<Conv2D> shortcut_;
};

class ResNet18 : public Module {
public:
    ResNet18(Runtime& rt, int num_classes = 1000);

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    Conv2D conv1_;
    BatchNorm bn1_;

    BasicBlock layer1_b1_;
    BasicBlock layer1_b2_;
    BasicBlock layer2_b1_;
    BasicBlock layer2_b2_;
    BasicBlock layer3_b1_;
    BasicBlock layer3_b2_;
    BasicBlock layer4_b1_;
    BasicBlock layer4_b2_;

    Linear linear_;
};

} // namespace axon
