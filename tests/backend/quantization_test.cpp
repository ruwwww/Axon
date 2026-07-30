#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("quantized_size computes correct sizes", "[quant]") {
    REQUIRE(cpu::quantized_size(32, QuantFormat::Q8_0) == 34);
    REQUIRE(cpu::quantized_size(64, QuantFormat::Q8_0) == 68);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q8_0) == 34);
    REQUIRE(cpu::quantized_size(33, QuantFormat::Q8_0) == 68);

    REQUIRE(cpu::quantized_size(32, QuantFormat::Q4_0) == 18);
    REQUIRE(cpu::quantized_size(64, QuantFormat::Q4_0) == 36);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q4_0) == 18);
    REQUIRE(cpu::quantized_size(33, QuantFormat::Q4_0) == 36);

    // K-quant formats use 256-element blocks
    REQUIRE(cpu::quantized_size(256, QuantFormat::Q2_K) == 84);
    REQUIRE(cpu::quantized_size(512, QuantFormat::Q2_K) == 168);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q2_K) == 84);

    REQUIRE(cpu::quantized_size(256, QuantFormat::Q3_K) == 110);
    REQUIRE(cpu::quantized_size(512, QuantFormat::Q3_K) == 220);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q3_K) == 110);

    REQUIRE(cpu::quantized_size(256, QuantFormat::Q4_K) == 144);
    REQUIRE(cpu::quantized_size(512, QuantFormat::Q4_K) == 288);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q4_K) == 144);

    REQUIRE(cpu::quantized_size(256, QuantFormat::Q5_K) == 176);
    REQUIRE(cpu::quantized_size(512, QuantFormat::Q5_K) == 352);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q5_K) == 176);

    REQUIRE(cpu::quantized_size(256, QuantFormat::Q6_K) == 210);
    REQUIRE(cpu::quantized_size(512, QuantFormat::Q6_K) == 420);
    REQUIRE(cpu::quantized_size(1, QuantFormat::Q6_K) == 210);
}

static void test_kquant_roundtrip(QuantFormat fmt, size_t count, float margin, int block_size, size_t expected_qsize) {
    Runtime rt;
    auto src = Tensor::randn(rt, {static_cast<int64_t>(count)});
    float* src_data = src.data<float>();
    for (size_t i = 0; i < count; ++i) {
        int64_t mid = static_cast<int64_t>(count) / 2;
        float v = static_cast<float>(static_cast<int64_t>(i) - mid) * 0.5f;
        // Alternate sign every 8 elements so each 16-element sub-block crosses zero
        // (K-quants can't represent all-positive/all-negative sub-blocks)
        src_data[i] = ((i / 8) % 2 == 0) ? v : -v;
    }

    size_t qsize = cpu::quantized_size(count, fmt);
    REQUIRE(qsize == expected_qsize);
    auto qtype = TensorType::contiguous({static_cast<int64_t>(count)}, DType::Float32, Device::CPU);
    StoragePtr qstorage = std::make_shared<Storage>(qsize);
    qstorage->quant = QuantizationDescriptor{fmt, block_size};
    Tensor quantized(qtype, qstorage, false);

    auto q_result = cpu::quantize(quantized, src, fmt);
    REQUIRE(q_result);

    auto dst = Tensor::empty(rt, {static_cast<int64_t>(count)});
    auto dq_result = cpu::dequantize(dst, quantized);
    REQUIRE(dq_result);

    float* dst_data = dst.data<float>();
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(dst_data[i] == Catch::Approx(src_data[i]).margin(margin));
    }
}

TEST_CASE("Q8_0 quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q8_0, 64, 0.5f, 32, 68);
}

TEST_CASE("Q4_0 quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q4_0, 128, 4.0f, 32, 72);
}

TEST_CASE("Q2_K quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q2_K, 256, 10.0f, 256, 84);
}

TEST_CASE("Q3_K quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q3_K, 256, 15.0f, 256, 110);
}

TEST_CASE("Q4_K quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q4_K, 256, 8.0f, 256, 144);
}

TEST_CASE("Q5_K quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q5_K, 256, 5.0f, 256, 176);
}

TEST_CASE("Q6_K quantize-dequantize roundtrip preserves values", "[quant]") {
    test_kquant_roundtrip(QuantFormat::Q6_K, 256, 4.0f, 256, 210);
}

static void test_kquant_matmul(QuantFormat fmt, int64_t M, int64_t K, int64_t N, float margin, int64_t qblock) {
    Runtime rt;
    auto a_f32 = Tensor::randn(rt, {M, K});
    auto b_f32 = Tensor::randn(rt, {K, N});

    auto ref_type = TensorType::contiguous({M, N}, DType::Float32);
    Tensor ref(ref_type, rt.allocator().allocate(ref_type), false);
    cpu::matmul(ref, a_f32, b_f32);

    size_t a_qsize = cpu::quantized_size_2d(M, K, fmt);
    StoragePtr a_qstorage = std::make_shared<Storage>(a_qsize);
    a_qstorage->quant = QuantizationDescriptor{fmt, qblock};
    Tensor a_q(TensorType::contiguous({M, K}, DType::Float32, Device::CPU), a_qstorage, false);
    REQUIRE(cpu::quantize(a_q, a_f32, fmt));

    auto out_type = TensorType::contiguous({M, N}, DType::Float32);
    Tensor out(out_type, rt.allocator().allocate(out_type), false);

    Expected<void> mm_result = Error{"test_kquant_matmul: unsupported format"};
    if (fmt == QuantFormat::Q2_K) {
        mm_result = cpu::matmul_q2_K(out, a_q, b_f32);
    } else if (fmt == QuantFormat::Q3_K) {
        mm_result = cpu::matmul_q3_K(out, a_q, b_f32);
    } else if (fmt == QuantFormat::Q4_K) {
        mm_result = cpu::matmul_q4_K(out, a_q, b_f32);
    } else if (fmt == QuantFormat::Q5_K) {
        mm_result = cpu::matmul_q5_K(out, a_q, b_f32);
    } else if (fmt == QuantFormat::Q6_K) {
        mm_result = cpu::matmul_q6_K(out, a_q, b_f32);
    }
    REQUIRE(mm_result);

    float* ref_data = ref.data<float>();
    float* out_data = out.data<float>();
    for (int64_t i = 0; i < M * N; ++i) {
        REQUIRE(out_data[i] == Catch::Approx(ref_data[i]).margin(margin));
    }
}

TEST_CASE("Q4_K matmul matches float32 matmul within tolerance", "[quant]") {
    test_kquant_matmul(QuantFormat::Q4_K, 4, 256, 2, 8.0f, 256);
}

TEST_CASE("Q6_K matmul matches float32 matmul within tolerance", "[quant]") {
    test_kquant_matmul(QuantFormat::Q6_K, 4, 256, 2, 8.0f, 256);
}

TEST_CASE("Q2_K matmul matches float32 matmul within tolerance", "[quant]") {
    test_kquant_matmul(QuantFormat::Q2_K, 4, 256, 2, 15.0f, 256);
}

TEST_CASE("Q3_K matmul matches float32 matmul within tolerance", "[quant]") {
    test_kquant_matmul(QuantFormat::Q3_K, 4, 256, 2, 20.0f, 256);
}

TEST_CASE("Q5_K matmul matches float32 matmul within tolerance", "[quant]") {
    test_kquant_matmul(QuantFormat::Q5_K, 4, 256, 2, 8.0f, 256);
}

TEST_CASE("Q4_0 matmul matches float32 matmul within tolerance", "[quant]") {

    Runtime rt;
    auto a_f32 = Tensor::randn(rt, {4, 8});
    auto b_f32 = Tensor::randn(rt, {8, 2});

    auto ref_type = TensorType::contiguous({4, 2}, DType::Float32);
    Tensor ref(ref_type, rt.allocator().allocate(ref_type), false);
    cpu::matmul(ref, a_f32, b_f32);

    size_t a_qsize = cpu::quantized_size_2d(4, 8, QuantFormat::Q4_0);
    StoragePtr a_qstorage = std::make_shared<Storage>(a_qsize);
    a_qstorage->quant = QuantizationDescriptor{QuantFormat::Q4_0, 32};
    Tensor a_q(TensorType::contiguous({4, 8}, DType::Float32, Device::CPU), a_qstorage, false);
    REQUIRE(cpu::quantize(a_q, a_f32, QuantFormat::Q4_0));

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

    auto ref_type = TensorType::contiguous({2, 2}, DType::Float32);
    Tensor ref(ref_type, rt.allocator().allocate(ref_type), false);
    cpu::matmul(ref, a_f32, b_f32);

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
