#include "nccl_backend.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>

nccl_backend::nccl_backend() : nccl_comm_(nullptr) {}

nccl_backend::~nccl_backend() {
    if (nccl_comm_) {
        ncclCommDestroy((ncclComm_t)nccl_comm_);
    }
}

void nccl_backend::replay(const log_data& log, int iterations, bool verbose,
                          const buffer_pool& pool) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclCommInitRank(&comm, num_ranks, id, my_rank);
    nccl_comm_ = (void*)comm;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<std::vector<const exchange_entry*>> by_group(log.max_group_id);
    for (const auto& entry : log.entries) {
        by_group[entry.group_id - 1].push_back(&entry);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& entries : by_group) {
            if (entries.empty()) continue;

            const int gid = entries.front()->group_id;
            int buf_idx = 0;

            auto start = std::chrono::high_resolution_clock::now();

            ncclGroupStart();

            for (const auto* entry : entries) {
                if (entry->dir == direction::recv) {
                    ncclRecv(pool.buffers[buf_idx], entry->size_bytes, ncclInt8, entry->peer, comm,
                             stream);
                    ++buf_idx;
                }
            }

            for (const auto* entry : entries) {
                if (entry->dir == direction::send) {
                    ncclSend(pool.buffers[buf_idx], entry->size_bytes, ncclInt8, entry->peer, comm,
                             stream);
                    ++buf_idx;
                }
            }

            ncclGroupEnd();
            cudaStreamSynchronize(stream);

            auto end = std::chrono::high_resolution_clock::now();
            int64_t duration_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            timings_.push_back({my_rank, gid, iter, duration_ns});
        }
    }

    ncclCommDestroy(comm);
    nccl_comm_ = nullptr;
    cudaStreamDestroy(stream);
}

std::vector<timing_result> nccl_backend::get_timings() const { return timings_; }
