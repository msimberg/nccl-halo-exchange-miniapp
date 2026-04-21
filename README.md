# NCCL/MPI Halo Exchange Replay Miniapp

Replays halo exchange communication patterns from a log file using either MPI or NCCL backends for performance comparison.

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
mpirun -n 4 ./halo_replay --log <path> --backend <mpi|nccl> [--iterations N] [--verbose]
```

### Options

- `--log <path>`: Path to halo exchange log file (required)
- `--backend <name>`: Backend to use: `mpi` or `nccl` (default: mpi)
- `--iterations <N>`: Number of times to replay the log (default: 1)
- `--verbose`: Output per-exchange CSV timings to stdout

### Log Format

CSV with columns: `timestamp_ns,rank,comm,group_id,direction,peer,size_bytes`

Example:
```
timestamp_ns,rank,comm,group_id,direction,peer,size_bytes
116390742302564,1,0x7ffe17d49b08,1,send,2,4000
116390742354109,1,0x7ffe17d49b08,1,recv,0,4000
```

## Output

### Verbose Mode

CSV format: `rank,group_id,exchange_num,timestamp_ns,duration_ns`

### Summary

At the end of execution, prints min/max/avg total time across all ranks.
