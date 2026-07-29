#pragma once

#include <string>
#include "axon/core/expected.h"
#include "axon/nn/module.h"
#include "axon/runtime/runtime.h"
#include "axon/tensor/tensor.h"

namespace axon {

Expected<void> save_tensor(const Tensor& t, const std::string& path);
Expected<Tensor> load_tensor(Runtime& rt, const std::string& path);

Expected<void> save_checkpoint(const Module& m, const std::string& path);
Expected<void> load_checkpoint(Runtime& rt, Module& m, const std::string& path);

} // namespace axon
