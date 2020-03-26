/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/src/cudnn/cudnn_all_server.cc
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Monday, December 10th 2018, 6:28:34 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Sunday, January 27th 2019, 9:41:12 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
    DLOG(INFO) << "Process " << cur_pid <<" is at line " << __LINE__;
 */

#include <pthread.h>
#include <boost/program_options.hpp>
#include "cudnn/cudnn_server.hpp"

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
        "percentage of cudnn resource usage");
    boost::program_options::variables_map var_map;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, opts),
        var_map);
    boost::program_options::notify(var_map);

    DLOG(INFO) << "Server started, Using " << cur_percent << "% sources!";

    cur_percent = cur_percent / 10;
    DescriptorStore desc_store;
    desc_store.init();
    DeviceStore dev_store;
    // DescriptorStore *new_store = &descriptor_store;

    // SetArgs set_args;
    // ComputeArgs compute_args;

    // set_args.descriptor_store = &descriptor_store;

    // compute_args.device_store = &device_store;
    // compute_args.descriptor_store = &descriptor_store;
    // compute_args.cur_percent_ = &cur_percent;
    // compute_args.cur_pid_ = &cur_pid;

    // map shared memory for set
    int fd = shm_open(getenv("SHARED_SET_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    ShmSet *shm_set = (ShmSet *)mmap(NULL, SHM_SET_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    int ret = close(fd);
    CHECK_NE(ret, -1);
    // map shared memory for compute
    fd = shm_open(getenv("SHARED_COMPUTE_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    ShmCompute *shm_compute = (ShmCompute *)mmap(NULL, SHM_COMPUTE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ret = close(fd);
    CHECK_NE(ret, -1);
    DLOG(INFO) << "Reading All Shared Memory Succeed!";

    // cudaEvent_t complete;
    // cudaEvent_t start;
    cudnnHandle_t cudnn;
    // CUDA_CHECK(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming | cudaEventInterprocess));
    CUDNN_CHECK(cudnnCreate(&cudnn));
    api_server(&cudnn, shm_compute, shm_set, desc_store, dev_store, cur_percent, cur_pid);
    // compute_args.complete_ = &complete;
    // compute_args.start_ = &start;
    // compute_args.cudnn = &cudnn;


    // pthread_t set;
    // pthread_t compute;

    // pthread_create(&set, NULL, set_server, (void *)&set_args);
    // pthread_create(&compute, NULL, compute_server, (void *)&compute_args);
    // // pthread_create(&compute, NULL, compute_server_test, (void *)&compute_args);
    // pthread_join(set, NULL);
    // pthread_join(compute, NULL);
    return 0;
}