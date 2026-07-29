#include <fstream>
#include <vector>
#include <cstring>
#include "axon/core/serialize.h"

namespace axon {

namespace {

constexpr uint32_t AXON_MAGIC = 0x4E4F5841; // "AXON" in little-endian
constexpr uint32_t AXON_VERSION = 1;

struct TensorHeader {
    uint32_t name_len;
    uint32_t dtype;
    uint32_t quant_type;
    uint32_t ndims;
};

Expected<void> write_tensor_to_stream(std::ofstream& f, const Tensor& t, const std::string& name) {
    auto& type = t.type();
    uint32_t name_len = static_cast<uint32_t>(name.size());
    uint32_t dtype = static_cast<uint32_t>(type.dtype());
    uint32_t quant_type = static_cast<uint32_t>(type.quant());
    uint32_t ndims = static_cast<uint32_t>(type.ndim());

    TensorHeader h{name_len, dtype, quant_type, ndims};
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!f) return Error{"Failed to write tensor header"};

    f.write(name.data(), name.size());
    if (!f) return Error{"Failed to write tensor name"};

    for (auto s : type.shape()) {
        uint64_t dim = static_cast<uint64_t>(s);
        f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        if (!f) return Error{"Failed to write shape dimension"};
    }

    uint64_t data_bytes = t.storage()->size_bytes;
    f.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
    if (!f) return Error{"Failed to write data_bytes"};

    f.write(static_cast<const char*>(t.storage()->data), static_cast<std::streamsize>(data_bytes));
    if (!f) return Error{"Failed to write tensor data"};

    return {};
}

Expected<void> read_tensor_header(std::ifstream& f, TensorHeader& h, std::string& name,
                                   std::vector<uint64_t>& shape, uint64_t& data_bytes) {
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) return Error{"Failed to read tensor header or unexpected end of file"};

    name.resize(h.name_len);
    f.read(name.data(), h.name_len);
    if (!f) return Error{"Failed to read tensor name"};

    shape.resize(h.ndims);
    for (uint32_t i = 0; i < h.ndims; ++i) {
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(uint64_t));
        if (!f) return Error{"Failed to read shape dimension"};
    }

    f.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));
    if (!f) return Error{"Failed to read data_bytes"};
    return {};
}

} // anonymous namespace

Expected<void> save_tensor(const Tensor& t, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return Error{"Failed to open file for writing: " + path};

    f.write(reinterpret_cast<const char*>(&AXON_MAGIC), sizeof(AXON_MAGIC));
    f.write(reinterpret_cast<const char*>(&AXON_VERSION), sizeof(AXON_VERSION));

    uint32_t count = 1;
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!f) return Error{"Failed to write tensor count"};

    return write_tensor_to_stream(f, t, "data");
}

Expected<Tensor> load_tensor(Runtime& rt, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Error{"Failed to open file for reading: " + path};

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!f || magic != AXON_MAGIC) return Error{"Invalid .axon file: bad magic"};

    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || version != AXON_VERSION) return Error{"Unsupported .axon version"};

    uint32_t count;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f || count < 1) return Error{"No tensors in file"};

    TensorHeader h;
    std::string name;
    std::vector<uint64_t> shape_u64;
    uint64_t data_bytes;

    auto err = read_tensor_header(f, h, name, shape_u64, data_bytes);
    if (!err) return err.error();

    std::vector<int64_t> shape(shape_u64.begin(), shape_u64.end());
    auto dtype = static_cast<DType>(h.dtype);

    auto type = TensorType::contiguous(std::move(shape), dtype, Device::CPU);
    auto storage = rt.allocator().allocate(type);

    if (data_bytes != storage->size_bytes) {
        return Error{"Tensor data size mismatch"};
    }

    f.read(static_cast<char*>(storage->data), static_cast<std::streamsize>(data_bytes));
    if (!f) return Error{"Failed to read tensor data"};

    return Tensor(std::move(type), std::move(storage), false);
}

Expected<void> save_checkpoint(const Module& m, const std::string& path) {
    auto params = m.parameters();
    auto& names = m.parameter_names();

    if (params.size() != names.size()) {
        return Error{"Parameter count mismatch"};
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) return Error{"Failed to open file for writing: " + path};

    f.write(reinterpret_cast<const char*>(&AXON_MAGIC), sizeof(AXON_MAGIC));
    f.write(reinterpret_cast<const char*>(&AXON_VERSION), sizeof(AXON_VERSION));

    uint32_t count = static_cast<uint32_t>(params.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!f) return Error{"Failed to write parameter count"};

    for (size_t i = 0; i < params.size(); ++i) {
        auto err = write_tensor_to_stream(f, params[i]->tensor(), names[i]);
        if (!err) return err.error();
    }

    return {};
}

Expected<void> load_checkpoint(Runtime& rt, Module& m, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Error{"Failed to open file for reading: " + path};

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!f || magic != AXON_MAGIC) return Error{"Invalid .axon file: bad magic"};

    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || version != AXON_VERSION) return Error{"Unsupported .axon version"};

    uint32_t count;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f) return Error{"Failed to read tensor count"};

    auto params = m.parameters();
    for (uint32_t i = 0; i < count; ++i) {
        TensorHeader h;
        std::string name;
        std::vector<uint64_t> shape_u64;
        uint64_t data_bytes;

        auto err = read_tensor_header(f, h, name, shape_u64, data_bytes);
        if (!err) return err.error();

        if (i >= params.size()) {
            return Error{"Checkpoint has more tensors than module parameters"};
        }

        auto* param = params[i];
        auto& tensor = param->tensor();
        auto& param_type = tensor.type();

        std::vector<int64_t> shape(shape_u64.begin(), shape_u64.end());
        if (shape != param_type.shape()) {
            return Error{"Shape mismatch for parameter"};
        }
        if (static_cast<DType>(h.dtype) != param_type.dtype()) {
            return Error{"DType mismatch for parameter"};
        }

        if (data_bytes != tensor.storage()->size_bytes) {
            return Error{"Data size mismatch for parameter"};
        }

        f.read(static_cast<char*>(tensor.storage()->data), static_cast<std::streamsize>(data_bytes));
        if (!f) return Error{"Failed to read parameter data"};
    }

    return {};
}

} // namespace axon
