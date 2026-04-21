#include "mpi_backend.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
mpi_backend::mpi_backend() {}

mpi_backend::~mpi_backend() {}

void mpi_backend::replay(const log_data& log, int iterations, bool verbose,
                         const buffer_pool& pool) {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    std::vector<std::vector<const exchange_entry*>> by_group(log.max_group_id);
    for (const auto& entry : log.entries) {
        by_group[entry.group_id - 1].push_back(&entry);
    }

    size_t max_ops_per_group = 0;
    for (const auto& entries : by_group) {
        max_ops_per_group = std::max(max_ops_per_group, entries.size());
    }

    std::vector<MPI_Request> requests(max_ops_per_group);

    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& entries : by_group) {
            if (entries.empty()) continue;

            const int gid = entries.front()->group_id;
            int req_count = 0;

            for (const auto* entry : entries) {
                if (entry->dir == direction::recv) {
                    MPI_Irecv(pool.buffers[req_count], static_cast<int>(entry->size_bytes),
                              MPI_BYTE, entry->peer, gid, MPI_COMM_WORLD, &requests[req_count]);
                    ++req_count;
                }
            }

            for (const auto* entry : entries) {
                if (entry->dir == direction::send) {
                    MPI_Isend(pool.buffers[req_count], static_cast<int>(entry->size_bytes),
                              MPI_BYTE, entry->peer, gid, MPI_COMM_WORLD, &requests[req_count]);
                    ++req_count;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();

            MPI_Waitall(req_count, requests.data(), MPI_STATUSES_IGNORE);

            auto end = std::chrono::high_resolution_clock::now();
            int64_t duration_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            timings_.push_back({my_rank, gid, iter, duration_ns});
        }
    }
}

std::vector<timing_result> mpi_backend::get_timings() const { return timings_; }
