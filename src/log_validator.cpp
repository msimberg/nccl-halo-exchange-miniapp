#include "log_validator.hpp"

#include <algorithm>
#include <set>
#include <sstream>

validation_error validate_log(const log_data& full_log, int mpi_num_ranks) {
    if (full_log.entries.empty()) {
        return {false, "Log file contains no entries"};
    }

    std::set<int> unique_ranks;
    for (const auto& entry : full_log.entries) {
        unique_ranks.insert(entry.rank);
    }

    if (static_cast<int>(unique_ranks.size()) != mpi_num_ranks) {
        std::ostringstream oss;
        oss << "Log file contains " << unique_ranks.size() << " unique ranks, but MPI job has "
            << mpi_num_ranks << " ranks";
        return {false, oss.str()};
    }

    int expected = 0;
    for (int r : unique_ranks) {
        if (r != expected) {
            std::ostringstream oss;
            oss << "Ranks are not contiguous 0..N-1: expected rank " << expected
                << " but found rank " << r;
            return {false, oss.str()};
        }
        ++expected;
    }

    struct comm_key {
        int rank;
        int peer;
        int group_id;
        direction dir;
        size_t size_bytes;

        bool operator<(const comm_key& other) const {
            if (rank != other.rank) return rank < other.rank;
            if (peer != other.peer) return peer < other.peer;
            if (group_id != other.group_id) return group_id < other.group_id;
            if (dir != other.dir) return dir < other.dir;
            return size_bytes < other.size_bytes;
        }
    };

    std::vector<comm_key> sends;
    std::vector<comm_key> recvs;
    for (const auto& entry : full_log.entries) {
        comm_key key{entry.rank, entry.peer, entry.group_id, entry.dir, entry.size_bytes};
        if (entry.dir == direction::send) {
            sends.push_back(key);
        } else {
            recvs.push_back(key);
        }
    }

    std::set<comm_key> send_set(sends.begin(), sends.end());
    std::set<comm_key> recv_set(recvs.begin(), recvs.end());

    std::vector<std::string> errors;

    for (const auto& s : send_set) {
        comm_key matching_recv{s.peer, s.rank, s.group_id, direction::recv, s.size_bytes};
        if (recv_set.find(matching_recv) == recv_set.end()) {
            std::ostringstream oss;
            oss << "Send from rank " << s.rank << " to rank " << s.peer << " in group "
                << s.group_id << " (" << s.size_bytes << " bytes) has no matching recv";
            errors.push_back(oss.str());
        }
    }

    for (const auto& r : recv_set) {
        comm_key matching_send{r.peer, r.rank, r.group_id, direction::send, r.size_bytes};
        if (send_set.find(matching_send) == send_set.end()) {
            std::ostringstream oss;
            oss << "Recv on rank " << r.rank << " from rank " << r.peer << " in group "
                << r.group_id << " (" << r.size_bytes << " bytes) has no matching send";
            errors.push_back(oss.str());
        }
    }

    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "Communication pair validation failed:\n";
        for (const auto& err : errors) {
            oss << "  " << err << "\n";
        }
        return {false, oss.str()};
    }

    return {true, ""};
}
