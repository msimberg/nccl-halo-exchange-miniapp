#include "buffer_pool.hpp"
#include "log_parser.hpp"
#include "log_validator.hpp"
#include "mpi_backend.hpp"
#include "nccl_backend.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

struct config {
    std::string log_file;
    std::string backend = "mpi";
    std::string memory = "gpu";
    int iterations = 1;
    bool verbose = false;
};

config parse_args(int argc, char** argv) {
    config cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            cfg.log_file = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg.backend = argv[++i];
        } else if (strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            cfg.memory = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: halo_replay [options]\n"
                      << "  --log <path>       Path to halo exchange log file\n"
                      << "  --backend <name>   Backend: mpi or nccl (default: mpi)\n"
                      << "  --memory <type>    Buffer memory: host or gpu (default: gpu)\n"
                      << "  --iterations <N>   Number of replay iterations (default: 1)\n"
                      << "  --verbose          Output per-exchange timings\n"
                      << "  --help             Show this help\n";
            exit(0);
        }
    }
    return cfg;
}

static std::pair<int, size_t> compute_buffer_requirements(const log_data& log) {
    std::map<int, int> ops_per_group;
    size_t max_size = 0;
    for (const auto& entry : log.entries) {
        ops_per_group[entry.group_id]++;
        max_size = std::max(max_size, entry.size_bytes);
    }
    int max_ops = 0;
    for (const auto& [gid, count] : ops_per_group) {
        max_ops = std::max(max_ops, count);
    }
    if (max_ops == 0) max_ops = 1;
    if (max_size == 0) max_size = 1;
    return {max_ops, max_size};
}

struct group_stats {
    int group_id;
    int iteration;
    int64_t min_ns;
    int64_t max_ns;
    int64_t avg_ns;
    int64_t median_ns;
};

static std::vector<group_stats> compute_group_stats(const std::vector<timing_result>& all_timings) {
    std::map<std::pair<int, int>, std::vector<int64_t>> grouped;
    for (const auto& t : all_timings) {
        grouped[{t.group_id, t.iteration}].push_back(t.duration_ns);
    }

    std::vector<group_stats> stats;
    for (const auto& [key, durations] : grouped) {
        auto sorted = durations;
        std::sort(sorted.begin(), sorted.end());
        int64_t sum = std::accumulate(sorted.begin(), sorted.end(), int64_t{0});
        size_t n = sorted.size();
        int64_t median = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[n / 2];
        stats.push_back({key.first, key.second, sorted.front(), sorted.back(),
                         sum / static_cast<int64_t>(n), median});
    }
    std::sort(stats.begin(), stats.end(), [](const group_stats& a, const group_stats& b) {
        return std::tie(a.iteration, a.group_id) < std::tie(b.iteration, a.group_id);
    });
    return stats;
}

static std::vector<timing_result> gather_timings(const std::vector<timing_result>& local,
                                                 int my_rank, int num_ranks) {
    int timing_size = static_cast<int>(sizeof(timing_result));
    int local_bytes = static_cast<int>(local.size()) * timing_size;

    std::vector<int> byte_counts(num_ranks);
    MPI_Gather(&local_bytes, 1, MPI_INT, byte_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(num_ranks, 0);
    int total_bytes = 0;
    if (my_rank == 0) {
        for (int i = 0; i < num_ranks; ++i) {
            displs[i] = total_bytes;
            total_bytes += byte_counts[i];
        }
    }

    std::vector<timing_result> all;
    if (my_rank == 0) {
        all.resize(total_bytes / timing_size);
    }

    MPI_Gatherv(local.data(), local_bytes, MPI_BYTE, all.data(), byte_counts.data(), displs.data(),
                MPI_BYTE, 0, MPI_COMM_WORLD);

    return all;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    config cfg = parse_args(argc, argv);

    if (cfg.log_file.empty()) {
        if (my_rank == 0) {
            std::cerr << "Error: --log <path> is required\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (cfg.memory != "host" && cfg.memory != "gpu") {
        if (my_rank == 0) {
            std::cerr << "Error: --memory must be 'host' or 'gpu'\n";
        }
        MPI_Finalize();
        return 1;
    }

    log_data log;
    try {
        log = parse_log_file(cfg.log_file, my_rank);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << my_rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int validation_ok = 0;
    if (my_rank == 0) {
        try {
            log_data full_log = parse_log_file_all(cfg.log_file);
            auto result = validate_log(full_log, num_ranks);
            if (!result.is_valid) {
                std::cerr << "Validation error: " << result.message << "\n";
            } else {
                validation_ok = 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Rank 0: validation failed: " << e.what() << "\n";
        }
    }
    MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!validation_ok) {
        MPI_Finalize();
        return 1;
    }

    auto [max_ops, max_size] = compute_buffer_requirements(log);
    memory_type mem_type = (cfg.memory == "host") ? memory_type::host : memory_type::gpu;

    std::unique_ptr<buffer_pool> pool;
    try {
        pool = std::make_unique<buffer_pool>(max_ops, max_size, mem_type);
    } catch (const std::exception& e) {
        if (my_rank == 0) {
            std::cerr << "Error: buffer allocation failed: " << e.what() << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        std::cout << "Replaying " << log.entries.size() << " exchanges across " << num_ranks
                  << " ranks, " << cfg.iterations << " iteration(s)\n"
                  << "Backend: " << cfg.backend << "\n"
                  << "Memory: " << cfg.memory << "\n"
                  << "Buffers: " << max_ops << " x " << max_size << " bytes\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto total_start = std::chrono::high_resolution_clock::now();

    std::vector<timing_result> local_timings;

    if (cfg.backend == "mpi") {
        mpi_backend backend;
        backend.replay(log, cfg.iterations, cfg.verbose, *pool);
        local_timings = backend.get_timings();
    } else if (cfg.backend == "nccl") {
        nccl_backend backend;
        backend.replay(log, cfg.iterations, cfg.verbose, *pool);
        local_timings = backend.get_timings();
    } else {
        if (my_rank == 0) {
            std::cerr << "Error: Unknown backend '" << cfg.backend << "'\n";
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto total_end = std::chrono::high_resolution_clock::now();
    int64_t total_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count();

    int64_t min_time, max_time, sum_time;
    MPI_Reduce(&total_time_ns, &min_time, 1, MPI_INT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time_ns, &max_time, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time_ns, &sum_time, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    auto all_timings = gather_timings(local_timings, my_rank, num_ranks);

    if (my_rank == 0) {
        if (cfg.verbose) {
            std::cout << "rank,iteration,group_id,duration_ns\n";
            for (const auto& t : all_timings) {
                std::cout << t.rank << "," << t.iteration << "," << t.group_id << ","
                          << t.duration_ns << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "=== Summary ===\n"
                  << "Total time (min): " << min_time << " ns\n"
                  << "Total time (max): " << max_time << " ns\n"
                  << "Total time (avg): " << sum_time / num_ranks << " ns\n\n"
                  << "group_id,iteration,min_ns,max_ns,avg_ns,median_ns\n";
        for (const auto& s : compute_group_stats(all_timings)) {
            std::cout << s.group_id << "," << s.iteration << "," << s.min_ns << "," << s.max_ns
                      << "," << s.avg_ns << "," << s.median_ns << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
