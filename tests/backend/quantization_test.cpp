#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("quantized_size computes correct sizes", "[quant]") {
    REQUIRE(cpu::quantized_size(32, QuantFormat::Q8_0) == 34);   // 1 block: 2 + 32
    REQUIRE(cpu::quantized_size(64, QuantFormat::Q8_0) == 68);   // 2 blocks: 2*34
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q8_0) == 34);    // partial block still 34
    REQUIRE(cpu::quantized_size(33, QuantFormat::Q8_0) == 68);   // 2 blocks

    REQUIRE(cpu::quantized_size(32, QuantFormat::Q4_0) == 18);   // 1 block: 2 + 16
    REQUIRE(cpu::quantized_size(64, QuantFormat::Q4_0) == 36);   // 2 blocks
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q4_0) == 18);    // partial block
    REQUIRE(cpu::quantized_size(33, QuantFormat::Q4_0) == 36);   // 2 blocks
}

TEST_CASE("Q8_0 quantize-dequantize roundtrip preserves values", "[quant]") {
    Runtime rt;
    auto src = Tensor::randn(rt, {64});
    float* src_data = src.data<float>();
    for (int64_t i = 0; i < 64; ++i) src_data[i] = static_cast<float>(i) - 32.0f;

    size_t qsize = cpu::quantized_size(64, QuantFormat::Q8_0);
    auto qtype = TensorType::contiguous({64}, DType::Float32, Device::CPU);
    StoragePtr qstorage = std::make_shared<Storage>(qsize);
    qstorage->quant = QuantizationDescriptor{QuantFormat::Q8_0, 32};
    Tensor quantized(qtype, qstorage, false);

    auto q_result = cpu::quantize(quantized, src, QuantFormat::Q8_0);
    REQUIRE(q_result);

    auto dst = Tensor::empty(rt, {64});
    auto dq_result = cpu::dequantize(dst, quantized);
    REQUIRE(dq_result);

    float* dst_data = dst.data<float>();
    for (int64_t i = 0; i < 64; ++i) {
        REQUIRE(dst_data[i] == Catch::Approx(src_data[i]).margin(0.5f));
    }
}

TEST_CASE("Q4_0 quantize-dequantize roundtrip preserves values", "[quant]") {
    Runtime rt;
    auto src = Tensor::randn(rt, {128});
    float* src_data = src.data<float>();
    for (int64_t i = 0; i < 128; ++i) src_data[i] = static_cast<float>(i) * 0.1f;

    size_t qsize = cpu::quantized_size(128, QuantFormat::Q4_0);
    auto qtype = TensorType::contiguous({128}, DType::Float32, Device::CPU);
    StoragePtr qstorage = std::make_shared<Storage>(qsize);
    qstorage->quant = QuantizationDescriptor{QuantFormat::Q4_0, 32};
    Tensor quantized(qtype, qstorage, false);

    auto q_result = cpu::quantize(quantized, src, QuantFormat::Q4_0);
    REQUIRE(q_result);

    auto dst = Tensor::empty(rt, {128});
    auto dq_result = cpu::dequantize(dst, quantized);
    REQUIRE(dq_result);

    float* dst_data = dst.data<float>();
    for (int64_t i = 0; i < 128; ++i) {
        REQUIRE(dst_data[i] == Catch::Approx(src_data[i]).margin(2.0f));
    }
}

TEST_CASE("Q4_0 matmul matches float32 matmul within tolerance", "[quant]") {
    Runtime rt;
    // a: (4, 8), b: (8, 2) -> out: (4, 2)
    auto a_f32 = Tensor::randn(rt, {4, 8});
    auto b_f32 = Tensor::randn(rt, {8, 2});

    // Reference float32 matmul
    auto ref_type = TensorType::contiguous({4, 2}, DType::Float32);
    Tensor ref(ref_type, rt.allocator().allocate(ref_type), false);
    cpu::matmul(ref, a_f32, b_f32);

    // Quantize a to Q4_0 (per-row blocks for 2D)
    size_t a_qsize = cpu::quantized_size_2d(4, 8, QuantFormat::Q4_0);
    StoragePtr a_qstorage = std::make_shared<Storage>(a_qsize);
    a_qstorage->quant = QuantizationDescriptor{QuantFormat::Q4_0, 32};
    Tensor a_q(TensorType::contiguous({4, 8}, DType::Float32, Device::CPU), a_qstorage, false);
    REQUIRE(cpu::quantize(a_q, a_f32, QuantFormat::Q4_0));

    // Quantized matmul produces correct shape
    auto out_type = TensorType::contiguous({4, 2}, DType::Float32);
    Tensor out(out_type, rt.allocator().allocate(out_type), false);
    auto mm_result = cpu::matmul_q4(out, a_q, b_f32);
    REQUIRE(mm_result);

    float* ref_data = ref.data<float>();
    float* out_data = out.data<float>();
    for (int64_t i = 0; i < 8; ++i) {
        REQUIRE(out_data[i] == Catch::Approx(ref_data[i]).margin(1.5f));
    }
}

TEST_CASE("Q4_0 matmul with known values", "[quant]") {
    Runtime rt;
    auto a_f32 = rt.empty({2, 4});
    auto b_f32 = rt.empty({4, 2});
    float a_data[] = {1,2,3,4, 5,6,7,8};
    float b_data[] = {0.1f,0.2f, 0.3f,0.4f, 0.5f,0.6f, 0.7f,0.8f};
    memcpy(a_f32.data<float>(), a_data, 8 * sizeof(float));
    memcpy(b_f32.data<float>(), b_data, 8 * sizeof(float));

    // Reference float32 matmul
    auto ref_type = TensorType::contiguous({2, 2}, DType::Float32);
    Tensor ref(ref_type, rt.allocator().allocate(ref_type), false);
    cpu::matmul(ref, a_f32, b_f32);

    // Quantize a to Q4_0 (per-row blocks)
    size_t a_qsize = cpu::quantized_size_2d(2, 4, QuantFormat::Q4_0);
    StoragePtr a_qstorage = std::make_shared<Storage>(a_qsize);
    a_qstorage->quant = QuantizationDescriptor{QuantFormat::Q4_0, 32};
    Tensor a_q(TensorType::contiguous({2, 4}, DType::Float32, Device::CPU), a_qstorage, false);
    REQUIRE(cpu::quantize(a_q, a_f32, QuantFormat::Q4_0));

    auto out_type = TensorType::contiguous({2, 2}, DType::Float32);
    Tensor out(out_type, rt.allocator().allocate(out_type), false);
    REQUIRE(cpu::matmul_q4(out, a_q, b_f32));

    float* ref_data = ref.data<float>();
    float* out_data = out.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE(out_data[i] == Catch::Approx(ref_data[i]).margin(1.0f));
    }
}
