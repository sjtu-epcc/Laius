cmake_minimum_required(VERSION 3.0.0)

project(GPUBandwidth LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
#find_package(glog REQUIRED)
# find_package(boost REQUIRED)

set(CXX_FLAGS
    -g;
    -Wall;
    -O3;
    -std=c++11;
)

string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS}")
set(CMAKE_CXX_COMPILER "g++")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-limit=1000 -DNDEBUG")

set(GPUBandwidth_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(GPUBandwidth_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_CUDA_COMPILER "nvcc")
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# list(APPEND ARCH_FLAGS "--cudart shared")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
# message(STATUS ${CUDA_LIBARIES})

include_directories(
    ${GPUBandwidth_INCLUDE_DIR}
)
link_directories(
    /home/qchen/tools/lib
    /home/haohao/Tools/lib
    /home/whcui/tools/lib
    /usr/local/cuda/lib64
)

add_subdirectory(src/kernel)
add_subdirectory(src/naive)
add_subdirectory(src/optimized)
add_subdirectory(src/origin)
add_subdirectory(src/test)
