#include "buffer_pool.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>

buffer_pool::buffer_pool(int max_ops, size_t buf_size, memory_type type)
    : buffer_size(buf_size), mem_type(type) {
    buffers.resize(max_ops, nullptr);
    for (int i = 0; i < max_ops; ++i) {
        if (type == memory_type::gpu) {
            cudaError_t err = cudaMalloc(&buffers[i], buf_size);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed: " +
                                         std::string(cudaGetErrorString(err)));
            }
        } else {
            buffers[i] = std::malloc(buf_size);
            if (!buffers[i]) {
                throw std::runtime_error("malloc failed");
            }
        }
    }
}

buffer_pool::~buffer_pool() {
    for (auto* ptr : buffers) {
        if (!ptr) continue;
        if (mem_type == memory_type::gpu) {
            cudaFree(ptr);
        } else {
            std::free(ptr);
        }
    }
}
