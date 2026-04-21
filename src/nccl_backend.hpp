#pragma once

#include "buffer_pool.hpp"
#include "log_parser.hpp"
#include "mpi_backend.hpp"

#include <vector>

class nccl_backend {
  public:
    nccl_backend();
    ~nccl_backend();

    void replay(const log_data& log, int iterations, int warmup, bool verbose,
                const buffer_pool& pool);
    std::vector<timing_result> get_timings() const;
    int64_t get_total_time() const;
    int64_t get_warmup_time() const;

  private:
    std::vector<timing_result> timings_;
    void* nccl_comm_ = nullptr;
    void* stream_ = nullptr;
    int64_t total_time_ns_ = 0;
    int64_t warmup_time_ns_ = 0;
};
