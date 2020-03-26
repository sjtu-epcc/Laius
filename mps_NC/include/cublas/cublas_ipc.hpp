/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/include/cudnn/cudnn_ipc.hpp
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Friday, December 21st 2018, 3:23:52 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 6:15:37 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cublas_v2.h>
#include "cublas/check.hpp"

struct DeviceStore
{
    std::unordered_map<uintptr_t, void *> malloc_store;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
};
struct alignas(16) SgemmInfo
{
    float alpha,beta;
    cublasOperation_t transa, transb;
    int m, n, k, lda, ldb, ldc;
    std::uintptr_t A_devptr, B_devptr, C_devptr;
    cudaIpcMemHandle_t A_handle, B_handle, C_handle;
};
struct alignas(16) ShmCublas
{
    int self_percent;
    int percent_flag;
    struct alignas(16) cublasAPI
    {
        CUBLAS_TYPE api_type;
        SgemmInfo sgemm_info;
        double duration[11];
        double bandwidth[11];
    } cublas_call;
    struct alignas(16) SyncEnforce
    {
        pthread_mutex_t mutex;
        pthread_barrier_t barrier;
        sem_t sch_sem;
        sem_t percent_sem[11];
        sem_t if_sync;
        sem_t synced;
    } sync;
    void init()
    {
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&sync.mutex, &mutexattr);
        pthread_mutexattr_destroy(&mutexattr);
        pthread_barrierattr_t barrierattr;
        pthread_barrierattr_init(&barrierattr);
        pthread_barrierattr_setpshared(&barrierattr, PTHREAD_PROCESS_SHARED);
        pthread_barrier_init(&sync.barrier, &barrierattr, 2); //todo: nums of barriers
        pthread_barrierattr_destroy(&barrierattr);
        sem_init(&sync.sch_sem, 10, 0);
        sem_init(&sync.if_sync, 10, 0);
        sem_init(&sync.synced, 10, 0);
        for (auto sem_iter : sync.percent_sem)
        {
            sem_init(&sem_iter, 10, 0);
        }
    }
};

#define SHM_CUBLAS_SIZE sizeof(ShmCublas)
class CublasInterProcess
{
  public:
    ShmCublas *shm_cublas;
    int api_cnt;

  public:
    int self_percent;
    CublasInterProcess()
    {
        int fd = shm_open(getenv("SHARED_CUBLAS_FNAME"),
                          O_RDWR, S_IRUSR | S_IWUSR);
        assert(fd != -1);
        shm_cublas = (ShmCublas *)mmap(NULL, SHM_CUBLAS_SIZE,
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd, 0);
        shm_cublas->percent_flag = shm_cublas->self_percent = atoi(getenv("SELF_PERCENT"));
        api_cnt = 0;
        // CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming | cudaEventInterprocess));
        // CUDA_CHECK(cudaIpcGetEventHandle(&shm_compute->sync.start_handle, start));
    }
    void send_cublas(CUBLAS_TYPE &cublas_type_,
                     SgemmInfo &sgem_info_,
                     double bandwidth_[],
                     double duration_[])
    {
        api_cnt++;
        shm_cublas->cublas_call.api_type = cublas_type_;
        memcpy(shm_cublas->cublas_call.bandwidth, bandwidth_, 11 * sizeof(double));
        memcpy(shm_cublas->cublas_call.duration, duration_, 11 * sizeof(double));
        // std::cout << shm_compute->cudnn_call.bandwidth[10] << shm_compute->cudnn_call.duration[10] << std::endl;
        switch (cublas_type_)
        {
        case SGEMM:
        {
            shm_cublas->cublas_call.sgemm_info = sgem_info_;
            break;
        }
        }
        sem_post(&shm_cublas->sync.if_sync);
        sem_post(&shm_cublas->sync.sch_sem);
        // int shm_sem_val;
        // sem_getvalue(&shm_cublas->sync.sch_sem, &shm_sem_val);
        // fprintf(stderr, "process %d is at line %d\n", shm_sem_val, __LINE__);
        pthread_barrier_wait(&shm_cublas->sync.barrier);
    }
    ~CublasInterProcess()
    {
        memset(shm_cublas, 0, SHM_CUBLAS_SIZE);
        shm_cublas->init();
        int ret;
        ret = munmap(shm_cublas, SHM_CUBLAS_SIZE);
        assert(ret == 0);
    }
};