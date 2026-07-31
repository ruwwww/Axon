#include "axon/data/dataloader.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

namespace axon {

DataLoader::DataLoader(Dataset& dataset, size_t batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {}

std::vector<DataLoader::Batch> DataLoader::iter() {
    std::vector<size_t> indices(dataset_.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle_) {
        static std::mt19937 gen(42);
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    std::vector<Batch> batches;

    for (size_t start = 0; start < indices.size(); start += batch_size_) {
        size_t end = std::min(start + batch_size_, indices.size());
        size_t bs = end - start;

        auto [first_input, first_target] = dataset_.get(indices[start]);
        auto in_shape = first_input.type().shape();
        auto targ_shape = first_target.type().shape();

        std::vector<int64_t> batch_in_shape = {static_cast<int64_t>(bs)};
        batch_in_shape.insert(batch_in_shape.end(), in_shape.begin(), in_shape.end());

        std::vector<int64_t> batch_targ_shape = {static_cast<int64_t>(bs)};
        batch_targ_shape.insert(batch_targ_shape.end(), targ_shape.begin(), targ_shape.end());

        auto batch_in_type = TensorMetadata::contiguous(batch_in_shape, first_input.type().dtype());
        auto batch_in_storage = std::make_shared<Storage>(batch_in_type.size_bytes());
        Tensor batch_in(batch_in_type, batch_in_storage, false);

        auto batch_targ_type = TensorMetadata::contiguous(batch_targ_shape, first_target.type().dtype());
        auto batch_targ_storage = std::make_shared<Storage>(batch_targ_type.size_bytes());
        Tensor batch_targ(batch_targ_type, batch_targ_storage, false);

        size_t in_elem_size = 1;
        for (auto s : in_shape) in_elem_size *= static_cast<size_t>(s);
        size_t targ_elem_size = 1;
        for (auto s : targ_shape) targ_elem_size *= static_cast<size_t>(s);

        size_t in_bytes = in_elem_size * size_of(first_input.type().dtype());
        size_t targ_bytes = targ_elem_size * size_of(first_target.type().dtype());

        // Copy first element
        std::memcpy(batch_in.data<float>(), first_input.data<float>(), in_bytes);
        std::memcpy(batch_targ.data<int64_t>(), first_target.data<int64_t>(), targ_bytes);

        for (size_t j = 1; j < bs; ++j) {
            auto [input, target] = dataset_.get(indices[start + j]);
            std::memcpy(
                static_cast<char*>(batch_in.storage()->data) + j * in_bytes,
                input.data<float>(),
                in_bytes
            );
            std::memcpy(
                static_cast<char*>(batch_targ.storage()->data) + j * targ_bytes,
                target.data<int64_t>(),
                targ_bytes
            );
        }

        batches.push_back({std::move(batch_in), std::move(batch_targ)});
    }

    return batches;
}

} // namespace axon
