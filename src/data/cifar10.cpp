#include "axon/data/cifar10.h"
#include <cstring>
#include <fstream>
#include <vector>

namespace axon {

struct CIFAR10::Impl {
    Runtime& rt;
    std::vector<uint8_t> data;
    std::vector<int> labels;
    size_t num_samples;

    Impl(Runtime& rt_, const std::string& path, bool train)
        : rt(rt_), num_samples(0)
    {
        if (train) {
            // CIFAR-10: 5 training batches: data_batch_1.bin .. data_batch_5.bin
            for (int i = 1; i <= 5; ++i) {
                std::string batch_path = path + "/data_batch_" + std::to_string(i) + ".bin";
                load_batch(batch_path);
            }
        } else {
            std::string batch_path = path + "/test_batch.bin";
            load_batch(batch_path);
        }
    }

    void load_batch(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            return; // silently skip missing files
        }

        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Each sample: 1 byte label + 3072 bytes image (32*32*3)
        size_t samples_in_file = file_size / 3073;
        size_t old_size = data.size() / 3072;

        data.resize((old_size + samples_in_file) * 3072);
        labels.resize(old_size + samples_in_file);

        std::vector<uint8_t> buffer(3073);
        for (size_t i = 0; i < samples_in_file; ++i) {
            file.read(reinterpret_cast<char*>(buffer.data()), 3073);
            labels[old_size + i] = static_cast<int>(buffer[0]);
            std::memcpy(&data[(old_size + i) * 3072], &buffer[1], 3072);
        }

        num_samples = labels.size();
    }
};

CIFAR10::CIFAR10(Runtime& rt, const std::string& path, bool train)
    : impl_(std::make_unique<Impl>(rt, path, train)) {}

CIFAR10::~CIFAR10() = default;

size_t CIFAR10::size() const {
    return impl_->num_samples;
}

std::pair<Tensor, Tensor> CIFAR10::get(size_t index) {
    // Image: CHW format (3, 32, 32), normalized to [0,1]
    auto img_type = TensorType::contiguous({3, 32, 32}, DType::Float32);
    Tensor img(img_type, impl_->rt.allocator().allocate(img_type), false);

    auto* img_ptr = img.data<float>();
    auto* data_ptr = &impl_->data[index * 3072];

    for (int64_t i = 0; i < 3072; ++i) {
        img_ptr[i] = static_cast<float>(data_ptr[i]) / 255.0f;
    }

    // Label
    auto label_type = TensorType::contiguous({1}, DType::Int64);
    Tensor label(label_type, impl_->rt.allocator().allocate(label_type), false);
    label.data<int64_t>()[0] = static_cast<int64_t>(impl_->labels[index]);

    return {std::move(img), std::move(label)};
}

} // namespace axon
