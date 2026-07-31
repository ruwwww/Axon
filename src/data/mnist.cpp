#include "axon/data/mnist.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace axon {

struct MNIST::Impl {
    Runtime& rt;
    std::vector<uint8_t> images;
    std::vector<uint8_t> labels;
    size_t num_images;
    int64_t rows;
    int64_t cols;

    Impl(Runtime& runtime, const std::string& path, bool train)
        : rt(runtime), num_images(0), rows(0), cols(0)
    {
        std::string prefix = path + "/" + (train ? "train" : "t10k");

        // Read images file
        {
            std::string img_path = prefix + "-images-idx3-ubyte";
            std::ifstream f(img_path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Cannot open: " + img_path);
            }

            auto read_be_u32 = [&f]() -> uint32_t {
                uint8_t buf[4];
                f.read(reinterpret_cast<char*>(buf), 4);
                return (static_cast<uint32_t>(buf[0]) << 24) |
                       (static_cast<uint32_t>(buf[1]) << 16) |
                       (static_cast<uint32_t>(buf[2]) << 8)  |
                       (static_cast<uint32_t>(buf[3]));
            };

            uint32_t magic = read_be_u32();
            (void)magic;
            num_images = read_be_u32();
            rows = read_be_u32();
            cols = read_be_u32();

            size_t img_size = num_images * rows * cols;
            images.resize(img_size);
            f.read(reinterpret_cast<char*>(images.data()), img_size);
        }

        // Read labels file
        {
            std::string lbl_path = prefix + "-labels-idx1-ubyte";
            std::ifstream f(lbl_path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Cannot open: " + lbl_path);
            }

            auto read_be_u32 = [&f]() -> uint32_t {
                uint8_t buf[4];
                f.read(reinterpret_cast<char*>(buf), 4);
                return (static_cast<uint32_t>(buf[0]) << 24) |
                       (static_cast<uint32_t>(buf[1]) << 16) |
                       (static_cast<uint32_t>(buf[2]) << 8)  |
                       (static_cast<uint32_t>(buf[3]));
            };

            uint32_t magic = read_be_u32();
            (void)magic;
            uint32_t n = read_be_u32();

            labels.resize(n);
            f.read(reinterpret_cast<char*>(labels.data()), n);
        }
    }
};

MNIST::MNIST(Runtime& rt, const std::string& path, bool train)
    : impl_(std::make_unique<Impl>(rt, path, train)) {}

MNIST::~MNIST() = default;

size_t MNIST::size() const {
    return impl_->num_images;
}

std::pair<Tensor, Tensor> MNIST::get(size_t index) {
    // Image tensor: shape {784}, dtype Float32, normalized to [0,1]
    auto img_type = TensorMetadata::contiguous({impl_->rows * impl_->cols}, DType::Float32);
    auto img_storage = std::make_shared<Storage>(img_type.size_bytes());
    Tensor img_tensor(img_type, img_storage, false);
    auto* img_ptr = img_tensor.data<float>();

    size_t offset = index * impl_->rows * impl_->cols;
    for (int64_t i = 0; i < impl_->rows * impl_->cols; ++i) {
        img_ptr[i] = static_cast<float>(impl_->images[offset + i]) / 255.0f;
    }

    // Label tensor: shape {1}, dtype Int64
    auto lbl_type = TensorMetadata::contiguous({1}, DType::Int64);
    auto lbl_storage = std::make_shared<Storage>(lbl_type.size_bytes());
    Tensor lbl_tensor(lbl_type, lbl_storage, false);
    lbl_tensor.data<int64_t>()[0] = static_cast<int64_t>(impl_->labels[index]);

    return {img_tensor, lbl_tensor};
}

} // namespace axon
