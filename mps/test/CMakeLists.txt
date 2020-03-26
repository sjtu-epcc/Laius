cmake_minimum_required(VERSION 3.0.0)

project(LaiusTest LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
# find_package(glog REQUIRED)
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

set(LaiusTest_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
set(LaiusTest_SRC_DIR ${PROJECT_SOURCE_DIR})

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
    ${LaiusTest_INCLUDE_DIR}
)
link_directories(
    /home/laius/tools/lib
    /home/whcui/tools/lib
    /usr/local/cuda/lib64
)

cuda_add_executable(schedule schedule.cc)
target_link_libraries(schedule pthread dl rt glog)

cuda_add_library(cublas_hook SHARED cublas_hook.cc)
target_link_libraries(cublas_hook pthread dl rt glog)
target_compile_options(cublas_hook PUBLIC -fopenmp -shared )

cuda_add_executable(cublas_server cublas_server.cc cublas_api.cc)
target_link_libraries(cublas_server cublas pthread dl rt glog boost_program_options)

cuda_add_executable(init init.cc)
target_link_libraries(init dl rt pthread glog boost_program_options)
target_compile_options(init PUBLIC -fopenmp)