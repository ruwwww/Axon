#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include "axon/core/serialize.h"
#include "axon/nn/linear.h"
#include "axon/runtime/runtime.h"

using namespace axon;

namespace fs = std::filesystem;

TEST_CASE("save_tensor / load_tensor roundtrip float32", "[serialize]") {
    Runtime rt;
    auto t = Tensor::randn(rt, {2, 3});
    float* orig_data = t.data<float>();
    for (int64_t i = 0; i < 6; ++i) orig_data[i] = static_cast<float>(i);

    auto path = fs::temp_directory_path() / "axon_roundtrip_f32.axon";
    auto save_result = save_tensor(t, path.string());
    REQUIRE(save_result);

    auto loaded = load_tensor(rt, path.string());
    REQUIRE(loaded);
    REQUIRE((*loaded).type().shape() == std::vector<int64_t>({2, 3}));
    REQUIRE((*loaded).type().dtype() == DType::Float32);
    float* loaded_data = (*loaded).data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(loaded_data[i] == Catch::Approx(orig_data[i]));
    }
    fs::remove(path);
}

TEST_CASE("save_tensor / load_tensor roundtrip int32", "[serialize]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {4}, DType::Int32);
    int32_t* orig_data = t.data<int32_t>();
    for (int64_t i = 0; i < 4; ++i) orig_data[i] = static_cast<int32_t>(i * 10);

    auto path = fs::temp_directory_path() / "axon_roundtrip_i32.axon";
    auto save_result = save_tensor(t, path.string());
    REQUIRE(save_result);

    auto loaded = load_tensor(rt, path.string());
    REQUIRE(loaded);
    REQUIRE((*loaded).type().shape() == std::vector<int64_t>({4}));
    REQUIRE((*loaded).type().dtype() == DType::Int32);
    int32_t* loaded_data = (*loaded).data<int32_t>();
    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE(loaded_data[i] == orig_data[i]);
    }
    fs::remove(path);
}

TEST_CASE("save_tensor / load_tensor roundtrip 1D tensor", "[serialize]") {
    Runtime rt;
    auto t = Tensor::ones(rt, {5}, DType::Float32);

    auto path = fs::temp_directory_path() / "axon_roundtrip_1d.axon";
    REQUIRE(save_tensor(t, path.string()));

    auto loaded = load_tensor(rt, path.string());
    REQUIRE(loaded);
    REQUIRE((*loaded).type().shape() == std::vector<int64_t>({5}));
    float* data = (*loaded).data<float>();
    for (int64_t i = 0; i < 5; ++i) {
        REQUIRE(data[i] == Catch::Approx(1.0f));
    }
    fs::remove(path);
}

TEST_CASE("load_tensor returns error on missing file", "[serialize]") {
    Runtime rt;
    auto result = load_tensor(rt, "nonexistent_file.axon");
    REQUIRE_FALSE(result);
}

TEST_CASE("save_checkpoint / load_checkpoint roundtrip", "[serialize]") {
    Runtime rt;
    auto linear = Linear(rt, 4, 3);

    auto path = fs::temp_directory_path() / "axon_checkpoint.axon";
    REQUIRE(save_checkpoint(linear, path.string()));

    // Load into a fresh Linear instance
    auto linear2 = Linear(rt, 4, 3);
    REQUIRE(load_checkpoint(rt, linear2, path.string()));

    // Compare parameters bit-for-bit
    auto params = linear.parameters();
    auto params2 = linear2.parameters();
    REQUIRE(params.size() == params2.size());
    for (size_t i = 0; i < params.size(); ++i) {
        auto& t1 = params[i]->tensor();
        auto& t2 = params2[i]->tensor();
        REQUIRE(t1.type().shape() == t2.type().shape());
        REQUIRE(t1.type().dtype() == t2.type().dtype());
        auto* d1 = t1.data<float>();
        auto* d2 = t2.data<float>();
        size_t n = static_cast<size_t>(t1.type().numel());
        for (size_t j = 0; j < n; ++j) {
            REQUIRE(d1[j] == Catch::Approx(d2[j]));
        }
    }
    fs::remove(path);
}

TEST_CASE(".axon file format has correct magic and version", "[serialize]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {1, 1});
    auto path = fs::temp_directory_path() / "axon_format_check.axon";
    REQUIRE(save_tensor(t, path.string()));

    // Read raw bytes
    std::ifstream f(path.string(), std::ios::binary);
    REQUIRE(f.is_open());

    char magic[4];
    f.read(magic, 4);
    REQUIRE(magic[0] == 'A');
    REQUIRE(magic[1] == 'X');
    REQUIRE(magic[2] == 'O');
    REQUIRE(magic[3] == 'N');

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    REQUIRE(version == 1);

    f.close();
    fs::remove(path);
}
