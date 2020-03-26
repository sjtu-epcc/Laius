/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/src/test/test_server.cc
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Sunday, December 23rd 2018, 7:04:51 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 2:55:32 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <pthread.h>
#include <boost/program_options.hpp>
#include "cublas/cublas_server.hpp"

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
        "percentage of cublas resource usage");
    boost::program_options::variables_map var_map;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, opts),
        var_map);
    boost::program_options::notify(var_map);

    DLOG(INFO) << "Server started, Using " << cur_percent << "% sources!";

    cur_percent = cur_percent / 10;
    int fd = shm_open(getenv("SHARED_CUBLAS_FNAME"),
                      O_RDWR,
                      S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    // int ret = ftruncate(fd, SHM_CUBLAS_SIZE);
    ShmCublas *shm_cublas = (ShmCublas *)mmap(NULL,
                                              SHM_CUBLAS_SIZE,
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, 0);
    // shm_cublas->init();
    int ret = close(fd);
    CHECK_NE(ret, -1);
    // map shared memory for compute
    DLOG(INFO) << "Reading All Shared Memory Succeed!";

    DeviceStore dev_store;
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublas_api_server(&cublas_handle, shm_cublas, &dev_store, cur_percent, cur_pid);
    return 0;
}