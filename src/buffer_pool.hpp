#pragma once

#include <cstddef>
#include <vector>

enum class memory_type { host, gpu };

struct buffer_pool {
    std::vector<void*> buffers;
    size_t buffer_size;
    memory_type mem_type;

    buffer_pool(int max_ops, size_t buffer_size, memory_type type);
    ~buffer_pool();

    buffer_pool(const buffer_pool&) = delete;
    buffer_pool& operator=(const buffer_pool&) = delete;
};
