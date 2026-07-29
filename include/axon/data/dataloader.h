#pragma once

#include <cstddef>
#include <vector>
#include "axon/data/dataset.h"

namespace axon {

class DataLoader {
public:
    struct Batch {
        Tensor inputs;
        Tensor targets;
    };

    DataLoader(Dataset& dataset, size_t batch_size, bool shuffle = true);

    std::vector<Batch> iter();

private:
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
};

} // namespace axon
