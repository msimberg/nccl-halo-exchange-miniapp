#pragma once

#include "buffer_pool.hpp"
#include "log_parser.hpp"

#include <vector>

struct timing_result {
    int rank;
    int group_id;
    int iteration;
    int64_t duration_ns;
};

class mpi_backend {
  public:
    mpi_backend();
    ~mpi_backend();

    void replay(const log_data& log, int iterations, int warmup, bool verbose,
                const buffer_pool& pool);
    std::vector<timing_result> get_timings() const;

    int64_t get_total_time() const;
    int64_t get_warmup_time() const;

  private:
    std::vector<timing_result> timings_;
    int64_t total_time_ns_ = 0;
    int64_t warmup_time_ns_ = 0;
};
