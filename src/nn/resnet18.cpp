#include <cstring>
#include "axon/nn/resnet18.h"
#include "axon/runtime/runtime.h"

namespace axon {

// ── BasicBlock ──────────────────────────────────────────────────────────

BasicBlock::BasicBlock(Runtime& rt, int64_t in_planes, int64_t planes, int64_t stride)
    : conv1_(rt, in_planes, planes, 3, stride, 1, false)
    , bn1_(rt, planes)
    , conv2_(rt, planes, planes, 3, 1, 1, false)
    , bn2_(rt, planes)
{
    if (stride != 1 || in_planes != planes) {
        shortcut_ = std::make_unique<Conv2D>(rt, in_planes, planes, 1, stride, 0, false);
    }
}

Expected<Tensor> BasicBlock::forward(Runtime& rt, const Tensor& x) {
    auto out = conv1_.forward(rt, x);
    if (!out) return out.error();

    out = bn1_.forward(rt, out.value());
    if (!out) return out.error();

    out = rt.relu(out.value());
    if (!out) return out.error();

    out = conv2_.forward(rt, out.value());
    if (!out) return out.error();

    out = bn2_.forward(rt, out.value());
    if (!out) return out.error();

    Tensor shortcut_out;
    if (shortcut_) {
        auto sc = shortcut_->forward(rt, x);
        if (!sc) return sc.error();
        shortcut_out = sc.value();
    } else {
        shortcut_out = x;
    }

    auto added = rt.add(out.value(), shortcut_out);
    if (!added) return added.error();

    return rt.relu(added.value());
}

std::vector<Parameter*> BasicBlock::parameters() {
    std::vector<Parameter*> all;
    for (auto* p : conv1_.parameters()) all.push_back(p);
    for (auto* p : bn1_.parameters()) all.push_back(p);
    for (auto* p : conv2_.parameters()) all.push_back(p);
    for (auto* p : bn2_.parameters()) all.push_back(p);
    if (shortcut_) {
        for (auto* p : shortcut_->parameters()) all.push_back(p);
    }
    return all;
}

// ── ResNet18 ────────────────────────────────────────────────────────────

ResNet18::ResNet18(Runtime& rt, int num_classes)
    : conv1_(rt, 3, 64, 7, 2, 3, false)
    , bn1_(rt, 64)
    , layer1_b1_(rt, 64, 64, 1)
    , layer1_b2_(rt, 64, 64, 1)
    , layer2_b1_(rt, 64, 128, 2)
    , layer2_b2_(rt, 128, 128, 1)
    , layer3_b1_(rt, 128, 256, 2)
    , layer3_b2_(rt, 256, 256, 1)
    , layer4_b1_(rt, 256, 512, 2)
    , layer4_b2_(rt, 512, 512, 1)
    , linear_(rt, 512, num_classes)
{}

Expected<Tensor> ResNet18::forward(Runtime& rt, const Tensor& x) {
    auto out = conv1_.forward(rt, x);
    if (!out) return out.error();

    out = bn1_.forward(rt, out.value());
    if (!out) return out.error();

    out = rt.relu(out.value());
    if (!out) return out.error();

    out = rt.maxpool2d(out.value(), 3, 2);
    if (!out) return out.error();

    out = layer1_b1_.forward(rt, out.value());
    if (!out) return out.error();
    out = layer1_b2_.forward(rt, out.value());
    if (!out) return out.error();

    out = layer2_b1_.forward(rt, out.value());
    if (!out) return out.error();
    out = layer2_b2_.forward(rt, out.value());
    if (!out) return out.error();

    out = layer3_b1_.forward(rt, out.value());
    if (!out) return out.error();
    out = layer3_b2_.forward(rt, out.value());
    if (!out) return out.error();

    out = layer4_b1_.forward(rt, out.value());
    if (!out) return out.error();
    out = layer4_b2_.forward(rt, out.value());
    if (!out) return out.error();

    // Average pool (N, C, H, W) -> (N, C, 1, 1), then flatten -> (N, C)
    auto pooled = rt.avgpool2d(out.value(), 7, 1);
    if (!pooled) return pooled.error();

    auto& t = pooled.value();
    auto shape = t.type().shape();
    int64_t N = shape[0], C = shape[1];

    auto flat_type = TensorMetadata::contiguous({N, C}, DType::Float32);
    Tensor flat(flat_type, rt.allocator().allocate(flat_type), t.requires_grad());
    std::memcpy(flat.data<float>(), t.data<float>(), static_cast<size_t>(N * C) * 4);

    return linear_.forward(rt, flat);
}

std::vector<Parameter*> ResNet18::parameters() {
    std::vector<Parameter*> all;
    for (auto* p : conv1_.parameters()) all.push_back(p);
    for (auto* p : bn1_.parameters()) all.push_back(p);
    for (auto* p : layer1_b1_.parameters()) all.push_back(p);
    for (auto* p : layer1_b2_.parameters()) all.push_back(p);
    for (auto* p : layer2_b1_.parameters()) all.push_back(p);
    for (auto* p : layer2_b2_.parameters()) all.push_back(p);
    for (auto* p : layer3_b1_.parameters()) all.push_back(p);
    for (auto* p : layer3_b2_.parameters()) all.push_back(p);
    for (auto* p : layer4_b1_.parameters()) all.push_back(p);
    for (auto* p : layer4_b2_.parameters()) all.push_back(p);
    for (auto* p : linear_.parameters()) all.push_back(p);
    return all;
}

} // namespace axon
