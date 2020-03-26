/*
 * File: /home/haohao/Projects/Paper/reference/mps/src/test/test_server.cc
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Sunday, December 23rd 2018, 7:04:51 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, December 24th 2018, 4:04:35 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <pthread.h>
#include <boost/program_options.hpp>
#include "check.hpp"
#include "optimized/cuda_ipc.hpp"

int main(int argc, char const *argv[])
{
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    int cur_percent;
    int cur_pid = getpid();
    //add options
    boost::program_options::options_description opts("server options");
    opts.add_options()(
        "help,h", "help message")(
        "percent,p",
        boost::program_options::value<int>(&cur_percent)
            ->default_value(100)
            ->implicit_value(100),
        "percentage of mps resource usage");
    boost::program_options::variables_map var_map;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, opts),
        var_map);
    boost::program_options::notify(var_map);

    DLOG(INFO) << "Server started, Using " << cur_percent << "% sources!";
    int fd = shm_open(getenv("SHARED_MALLOC_FNAME"),
                      O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                      S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    int ret = ftruncate(fd, SHM_MALLOC_SIZE);
    CHECK_NE(ret, -1);
    ShmMalloc *shm_malloc = (ShmMalloc *)mmap(nullptr, SHM_MALLOC_SIZE,
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, 0);
    shm_malloc->init(1);
    while (true)
    {
        pthread_barrier_wait(&shm_malloc->sync.barrier);
        DLOG(INFO) << "Malloc received";
        void *tmp_ptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&tmp_ptr,
                                        shm_malloc->malloc_call.device_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
        CUDA_CHECK(cudaIpcCloseMemHandle(tmp_ptr));
        pthread_barrier_wait(&shm_malloc->sync.barrier);
    }
}