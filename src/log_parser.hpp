#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class direction { send, recv };

struct exchange_entry {
    int64_t timestamp_ns;
    int rank;
    int group_id;
    direction dir;
    int peer;
    size_t size_bytes;
};

struct log_data {
    std::vector<exchange_entry> entries;
    int num_ranks;
    int max_group_id;
};

log_data parse_log_file(const std::string& path, int my_rank);
log_data parse_log_file_all(const std::string& path);
