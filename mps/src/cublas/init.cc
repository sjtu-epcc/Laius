/*
 * File: /Users/gema/Projects/Laius/code/mps/src/schedule/init.cc
 * Project: /Users/gema/Projects/Laius/code/mps
 * Created Date: Saturday, January 12th 2019, 11:25:18 am
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 8:16:45 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

// #include "cuda/ipc.hpp"
#include <boost/program_options.hpp>
#include <glog/logging.h>
// #include "cudnn/cudnn_ipc.hpp"
#include "cublas/cublas_ipc.hpp"

int main(int argc, char const *argv[])
{
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    int num_of_processes;

    boost::program_options::options_description opts("initializer options");
    opts.add_options()(
        "help,h", "help message")(
        "process,p",
        boost::program_options::value<int>(&num_of_processes)
            ->default_value(1)
            ->implicit_value(1),
        "num of processes in process pool");
    boost::program_options::variables_map var_map;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, opts),
        var_map);
    boost::program_options::notify(var_map);

    DLOG(INFO) << "Setting of Num of Processes in Process Must be Correct! Current Num: " << num_of_processes;

    int fd = shm_open(getenv("SHARED_CUBLAS_FNAME"),
                  O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                  S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    int ret = ftruncate(fd, SHM_CUBLAS_SIZE);
    CHECK_NE(ret, -1);
    DLOG(INFO) << "Shared Memory for CUBLAS API Created Successfully!";
    ShmCublas *shm_cublas = (ShmCublas *)mmap(NULL, SHM_CUBLAS_SIZE,
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, 0);
    shm_cublas->init();
    ret = close(fd);
    CHECK_NE(ret, -1);
    ret = munmap(shm_cublas, SHM_CUBLAS_SIZE);
    CHECK_NE(ret, -1);
    return 0;
}

// int main()
// {
//     int ret;
//     int fd = shm_open(getenv("SHARED_MEMORY_FNAME"), O_RDWR | O_CREAT | O_TRUNC | O_EXCL, S_IRUSR | S_IWUSR);
//     assert(fd != -1);
//     ret = ftruncate(fd, SHARED_MEM_SIZE);
//     assert(ret != -1);
//     SharedMemoryContents *shared_memory = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE,
//                                                                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     shared_memory->init();
//     ret = close(fd);
//     assert(ret != -1);
//     ret = munmap(shared_memory, SHARED_MEM_SIZE);
//     assert(ret != -1);
//     exit(0);
// }
