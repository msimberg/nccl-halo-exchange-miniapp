#pragma once

#include "buffer_pool.hpp"
#include "log_parser.hpp"
#include "mpi_backend.hpp"

#include <vector>

class nccl_backend {
  public:
    nccl_backend();
    ~nccl_backend();

    void replay(const log_data& log, int iterations, bool verbose, const buffer_pool& pool);
    std::vector<timing_result> get_timings() const;

  private:
    std::vector<timing_result> timings_;
    void* nccl_comm_;
};
