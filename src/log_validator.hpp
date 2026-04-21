#pragma once

#include "log_parser.hpp"

#include <string>

struct validation_error {
    bool is_valid;
    std::string message;
};

validation_error validate_log(const log_data& full_log, int mpi_num_ranks);
