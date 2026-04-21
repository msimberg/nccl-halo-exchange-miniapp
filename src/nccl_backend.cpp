#include "nccl_backend.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>

nccl_backend::nccl_backend() : nccl_comm_(nullptr) {
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
    stream_ = (void*)stream;
}

nccl_backend::~nccl_backend() {
    if (stream_) {
        cudaStreamDestroy((cudaStream_t)stream_);
    }
    if (nccl_comm_) {
        ncclCommDestroy((ncclComm_t)nccl_comm_);
    }
}

void nccl_backend::replay(const log_data& log, int iterations, int warmup, bool verbose,
                          const buffer_pool& pool) {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    ncclComm_t comm = (ncclComm_t)nccl_comm_;
    cudaStream_t stream = (cudaStream_t)stream_;

    std::vector<std::vector<const exchange_entry*>> by_group(log.max_group_id);
    for (const auto& entry : log.entries) {
        by_group[entry.group_id - 1].push_back(&entry);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto warmup_start = std::chrono::high_resolution_clock::now();
    for (int w = 0; w < warmup; ++w) {
        for (const auto& entries : by_group) {
            if (entries.empty()) continue;

            int buf_idx = 0;

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
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto warmup_end = std::chrono::high_resolution_clock::now();
    warmup_time_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(warmup_end - warmup_start).count();

    auto iter_start = warmup_end;
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto iter_end = std::chrono::high_resolution_clock::now();
    total_time_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end - iter_start).count();
}

std::vector<timing_result> nccl_backend::get_timings() const { return timings_; }

int64_t nccl_backend::get_total_time() const { return total_time_ns_; }

int64_t nccl_backend::get_warmup_time() const { return warmup_time_ns_; }
