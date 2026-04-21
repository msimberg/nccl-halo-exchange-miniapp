#include "log_parser.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

static std::vector<exchange_entry> parse_raw(const std::string& path, std::set<int>& unique_ranks) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open log file: " + path);
    }

    std::vector<exchange_entry> all;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 7) continue;
        if (tokens[0] == "timestamp_ns") continue;

        exchange_entry entry;
        entry.timestamp_ns = std::stoll(tokens[0]);
        entry.rank = std::stoi(tokens[1]);
        entry.group_id = std::stoi(tokens[3]);
        entry.dir = (tokens[4] == "send") ? direction::send : direction::recv;
        entry.peer = std::stoi(tokens[5]);
        entry.size_bytes = std::stoull(tokens[6]);

        unique_ranks.insert(entry.rank);
        all.push_back(entry);
    }

    return all;
}

static log_data build_log_data(std::vector<exchange_entry> entries) {
    log_data data;
    data.entries = std::move(entries);
    data.max_group_id = 0;
    for (const auto& entry : data.entries) {
        if (entry.group_id > data.max_group_id) {
            data.max_group_id = entry.group_id;
        }
    }
    return data;
}

log_data parse_log_file_all(const std::string& path) {
    std::set<int> unique_ranks;
    auto all = parse_raw(path, unique_ranks);

    log_data data = build_log_data(std::move(all));
    data.num_ranks = static_cast<int>(unique_ranks.size());
    return data;
}

log_data parse_log_file(const std::string& path, int my_rank) {
    std::set<int> unique_ranks;
    auto all = parse_raw(path, unique_ranks);

    std::vector<exchange_entry> filtered;
    for (const auto& entry : all) {
        if (entry.rank == my_rank) {
            filtered.push_back(entry);
        }
    }

    log_data data = build_log_data(std::move(filtered));
    data.num_ranks = static_cast<int>(unique_ranks.size());
    return data;
}
