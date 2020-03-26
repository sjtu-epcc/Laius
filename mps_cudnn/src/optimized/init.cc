/*
 * File: /home/haohao/Projects/Paper/reference/mps/src/init/cudnn_all_init.cc
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Monday, December 10th 2018, 6:29:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, January 3rd 2019, 8:48:32 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "optimized/cudnn_ipc.hpp"

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
    //Shared Memory for Cudnn Compute API
    int ret;
    int fd = shm_open(getenv("SHARED_COMPUTE_FNAME"),
                      O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                      S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1) << "Fail to open shared memory for cudnn compute api";
    ret = ftruncate(fd, SHM_COMPUTE_SIZE);
    CHECK_NE(ret, -1) << "Fail to create shared memory for cudnn compute api";

    DLOG(INFO) << "Shared Memory for CUDNN Compute API Created Successfully!";
    ShmCompute *shm_compute = (ShmCompute *)mmap(NULL, SHM_COMPUTE_SIZE,
                                                 PROT_READ | PROT_WRITE,
                                                 MAP_SHARED, fd, 0);
    shm_compute->init();
    DLOG(INFO) << "All Shared Memory of CUDNN Compute API for Processes Initialized Successfully!";

    ret = close(fd);
    CHECK_NE(ret, -1) << "Fail to close fd of shared memory of cudnn compute api";
    ret = munmap(shm_compute, SHM_COMPUTE_SIZE);
    CHECK_NE(ret, -1) << "Fail to unmap shared memory of cudnn compute api";

    //Shared Memory for Cudnn Set API
    //initialize all shared memory of set api
    fd = shm_open(getenv("SHARED_SET_FNAME"),
                  O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                  S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1) << "Fail to open shared memory for cudnn set api";
    ret = ftruncate(fd, SHM_SET_SIZE);
    CHECK_NE(ret, -1) << "Fail to create shared memory for cudnn set api";
    DLOG(INFO) << "Shared Memory for CUDNN Set API Created Successfully!";
    ShmSet *shm_set = (ShmSet *)mmap(NULL, SHM_SET_SIZE,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED, fd, 0);
    shm_set->init(num_of_processes);
    DLOG(INFO) << "Shared Memory for CUDNN Set API Initialized Successfully!";

    ret = close(fd);
    CHECK_NE(ret, -1) << "Fail to close fd of shared memory of cudnn set api";
    ret = munmap(shm_set, SHM_SET_SIZE);
    CHECK_NE(ret, -1) << "Fail to unmap shared memory of cudnn set api";

    return 0;
}
