cmake_minimum_required(VERSION 3.18)
project(nccl_halo_exchange LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h)
find_library(NCCL_LIBRARY NAMES nccl)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found")
endif()

add_library(nccl::nccl INTERFACE IMPORTED)
target_include_directories(nccl::nccl INTERFACE ${NCCL_INCLUDE_DIR})
target_link_libraries(nccl::nccl INTERFACE ${NCCL_LIBRARY})

add_executable(halo_replay
    src/main.cpp
    src/buffer_pool.cpp
    src/log_parser.cpp
    src/log_validator.cpp
    src/mpi_backend.cpp
    src/nccl_backend.cpp
)

target_link_libraries(halo_replay PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    nccl::nccl
)
