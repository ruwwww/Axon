cmake_minimum_required(VERSION 3.22)
project(axon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AXON_USE_BLAS "Enable external BLAS backend integration" OFF)

# Fetch Catch2 v3 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE AXON_SOURCES src/*.cpp)

# Apply SIMD flags specifically to AVX2 translation units
file(GLOB_RECURSE AVX2_SOURCES src/backend/cpu/avx2/*.cpp)
if(AVX2_SOURCES)
    if(MSVC)
        set_source_files_properties(${AVX2_SOURCES} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(${AVX2_SOURCES} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
endif()

# Main library
add_library(axon ${AXON_SOURCES})

if(AXON_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_link_libraries(axon PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(axon PUBLIC AXON_HAS_BLAS=1)
endif()

# Test executable
enable_testing()
file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)

add_executable(axon_tests ${TEST_SOURCES})
target_link_libraries(axon_tests PRIVATE axon Catch2::Catch2WithMain)

# MNIST example
add_executable(mnist_example examples/mnist.cpp)
target_link_libraries(mnist_example PRIVATE axon)

# CIFAR10 example
add_executable(cifar10_example examples/cifar10.cpp)
target_link_libraries(cifar10_example PRIVATE axon)
