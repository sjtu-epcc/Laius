/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/cudnn_ipc.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Friday, December 21st 2018, 3:23:52 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 8th 2019, 10:56:26 pm
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
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cudnn.h>

#include "check.hpp"

struct MallocInfo
{
    void *device_ptr;
    cudaIpcMemHandle_t device_handle;
};

struct DeviceStore
{
    std::unordered_map<uintptr_t, MallocInfo> malloc_store;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
};

struct DescriptorStore
{
    std::unordered_map<uintptr_t, cudnnTensorDescriptor_t> tensor_store;
    std::unordered_map<uintptr_t, cudnnConvolutionDescriptor_t> conv_store;
    std::unordered_map<uintptr_t, cudnnFilterDescriptor_t> filter_store;
    std::unordered_map<uintptr_t, cudnnActivationDescriptor_t> activation_store;
    std::unordered_map<uintptr_t, cudnnPoolingDescriptor_t> pooling_store;
    pthread_mutex_t mutex;
    void init()
    {
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&mutex, &mutexattr);
        pthread_mutexattr_destroy(&mutexattr);
    }
};

struct alignas(16) TensorInfo
{
    // uintptr_t device_ptr;
    uintptr_t tensor_name;
    uintptr_t tensor_ptr;
    bool if_tensor_handle;
    cudaIpcMemHandle_t tensor_handle;
};

struct alignas(16) ConvFwdInfo
{
    uintptr_t filter_name;
    uintptr_t filter_ptr;
    bool if_filter_handle;
    cudaIpcMemHandle_t filter_handle;
    uintptr_t conv_name;
    uintptr_t workspace_ptr;
    bool if_workspace_handle;
    cudaIpcMemHandle_t workspace_handle;
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t algo;
};

struct alignas(16) ActivationFwdInfo
{
    uintptr_t activation_name;
};

struct alignas(16) PoolingFwdInfo
{
    uintptr_t pooling_name;
};

struct alignas(16) SoftmaxFwdInfo
{
    cudnnSoftmaxAlgorithm_t algo;
    cudnnSoftmaxMode_t mode;
};

struct alignas(16) ShmCompute
{
    volatile int percent_flag;
    volatile int process_flag;
    volatile bool if_computed;
    volatile bool if_sync;
    volatile float time_predict;;
    volatile float running_time[11]; 
    volatile int GPU_ratio;
    volatile float band_predict;
    volatile float sum_band;
    struct alignas(16) cudnnAPI
    {
        COMPUTE_TYPE api_type;
        float alpha, beta;
        TensorInfo tensor_in_info;
        TensorInfo tensor_out_info;
        ConvFwdInfo conv_fwd_info;
        ActivationFwdInfo activation_fwd_info;
        PoolingFwdInfo pooling_fwd_info;
        SoftmaxFwdInfo softmax_fwd_info;
        double duration[11];
        double bandwidth[11];
    } cudnn_call;
    struct alignas(16) SyncEnforce
    {
        pthread_mutex_t mutex;
        pthread_barrier_t barrier;
        // cudaIpcEventHandle_t complete_handle;
        // cudaIpcEventHandle_t start_handle;
    } sync;
    void init()
    {
        percent_flag = 100;
        process_flag = -1;
        if_computed = false;
        if_sync = false;
	time_predict = 0;
 	band_predict = 0;
	sum_band = 0;
	for(int i=0;i<11;i++)
		running_time[i] = 0;
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
    }
};

#define SHM_COMPUTE_SIZE sizeof(ShmCompute)
class ComputeInterProcess
{
  private:
    ShmCompute *shm_compute;
    int using_flag;

  public:
    ComputeInterProcess()
    {
        int fd = shm_open(getenv("SHARED_COMPUTE_FNAME"),
                          O_RDWR, S_IRUSR | S_IWUSR);
        assert(fd != -1);
        shm_compute = (ShmCompute *)mmap(NULL, SHM_COMPUTE_SIZE,
                                         PROT_READ | PROT_WRITE,
                                         MAP_SHARED, fd, 0);
        // CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming | cudaEventInterprocess));
        // CUDA_CHECK(cudaIpcGetEventHandle(&shm_compute->sync.start_handle, start));
    }
    void send_compute(COMPUTE_TYPE &api_type_,
                      float &alpha_, float &beta_,
                      TensorInfo &tensor_in_info_,
                      TensorInfo &tensor_out_info_,
                      ConvFwdInfo &conv_fwd_info_,
                      ActivationFwdInfo &activation_fwd_info_,
                      PoolingFwdInfo &pooling_fwd_info_,
                      SoftmaxFwdInfo &softmax_fwd_info_,
                      double bandwidth_[],
                      double duration_[])
    {
        // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        while (shm_compute->process_flag != -1)
            ;
        shm_compute->cudnn_call.api_type = api_type_;
        shm_compute->cudnn_call.alpha = alpha_;
        shm_compute->cudnn_call.beta = beta_;
        shm_compute->cudnn_call.tensor_in_info = tensor_in_info_;
        shm_compute->cudnn_call.tensor_out_info = tensor_out_info_;
        shm_compute->if_computed = false;
        memcpy(shm_compute->cudnn_call.bandwidth, bandwidth_, 11 * sizeof(double));
        memcpy(shm_compute->cudnn_call.duration, duration_, 11 * sizeof(double));
	shm_compute->time_predict += shm_compute->cudnn_call.duration[shm_compute->GPU_ratio];
	for(int i=1;i<11;i++){
		shm_compute->running_time[i] += shm_compute->cudnn_call.duration[shm_compute->GPU_ratio];
	}
        shm_compute->band_predict = shm_compute->cudnn_call.bandwidth[shm_compute->GPU_ratio];
	shm_compute->sum_band += shm_compute->band_predict;
        switch (api_type_)
        {
        case ADDTENSOR:
        {
            break;
        }
        case CONVFWD:
        {
            shm_compute->cudnn_call.conv_fwd_info = conv_fwd_info_;
            break;
        }
        case ACTIVATIONFWD:
        {
            shm_compute->cudnn_call.activation_fwd_info = activation_fwd_info_;
            break;
        }
        case POOLFWD:
        {
            shm_compute->cudnn_call.pooling_fwd_info = pooling_fwd_info_;
            break;
        }
        case SOFTMAXFWD:
        {
            shm_compute->cudnn_call.softmax_fwd_info = softmax_fwd_info_;
            break;
        }
        }
        shm_compute->process_flag = 0;
        CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shm_compute->sync.barrier); //event recorded
        // CUDA_CHECK(cudaIpcOpenEventHandle(&complete, shm_compute->sync.complete_handle));
        // CUDA_CHECK(cudaEventRecord(start));
        pthread_barrier_wait(&shm_compute->sync.barrier);
        // CUDA_CHECK(cudaEventSynchronize(complete)); //kernel complete
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    bool compute_need()
    {
        return !shm_compute->if_computed;
    }
    ~ComputeInterProcess()
    {
        memset(shm_compute, 0, SHM_COMPUTE_SIZE);
        shm_compute->init();
        int ret;
        ret = munmap(shm_compute, SHM_COMPUTE_SIZE);
        assert(ret == 0);
    }
};

struct alignas(16) SetTensor4d
{
    uintptr_t desc_name;
    cudnnTensorFormat_t format;
    cudnnDataType_t dataType;
    int n, c, h, w;
};

struct alignas(16) SetTensor4dex
{
    uintptr_t desc_name;
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
};

struct alignas(16) SetConv2d
{
    uintptr_t desc_name;
    int pad_h, pad_w, u, v, dilation_h, dilation_w;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t computeType;
};

struct alignas(16) SetFilter4d
{
    uintptr_t desc_name;
    cudnnTensorFormat_t format;
    cudnnDataType_t dataType;
    int k, c, h, w;
};

struct alignas(16) SetActivation
{
    uintptr_t desc_name;
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t reluNanOpt;
    double coef;
};

struct alignas(16) SetPooling2d
{
    uintptr_t desc_name;
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    int windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride;
};

struct alignas(16) ShmSet
{
    struct alignas(16) setAPI
    {
        SET_TYPE api_type;
        SetTensor4d set_tensor4d;
        SetTensor4dex set_tensor4dex;
        SetConv2d set_conv2d;
        SetFilter4d set_filter4d;
        SetActivation set_activation;
        SetPooling2d set_pooling2d;
    } set_call;
    struct alignas(16) SyncEnforce
    {
        pthread_mutex_t mutex;
        pthread_barrier_t barrier;
    } sync;
    void init(int num_of_process)
    {
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&sync.mutex, &mutexattr);
        pthread_mutexattr_destroy(&mutexattr);
        pthread_barrierattr_t barrierattr;
        pthread_barrierattr_init(&barrierattr);
        pthread_barrierattr_setpshared(&barrierattr, PTHREAD_PROCESS_SHARED);
        pthread_barrier_init(&sync.barrier, &barrierattr, num_of_process + 1); //todo: nums of barriers
        pthread_barrierattr_destroy(&barrierattr);
    }
};

#define SHM_SET_SIZE sizeof(ShmSet)

class SetInterProcess
{
  private:
    ShmSet *shm_set;

  public:
    SetInterProcess()
    {
        int fd = shm_open(getenv("SHARED_SET_FNAME"),
                          O_RDWR, S_IRUSR | S_IWUSR);
        assert(fd != -1);
        shm_set = (ShmSet *)mmap(NULL, SHM_SET_SIZE,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
    }
    void send_set(SET_TYPE &api_type_,
                  SetTensor4d &set_tensor4d_,
                  SetTensor4dex &set_tensor4dex_,
                  SetConv2d &set_conv2d_,
                  SetFilter4d &set_filter4d_,
                  SetActivation &set_activation_,
                  SetPooling2d &set_pooling2d_)
    {
        pthread_mutex_lock(&shm_set->sync.mutex);
        shm_set->set_call.api_type = api_type_;
        switch (api_type_)
        {
        case SETTENSOR4D:
        {
            shm_set->set_call.set_tensor4d = set_tensor4d_;
            break;
        }
        case SETTENSOR4DEX:
        {
            shm_set->set_call.set_tensor4dex = set_tensor4dex_;
            break;
        }
        case SETCONV2D:
        {
            shm_set->set_call.set_conv2d = set_conv2d_;
            break;
        }
        case SETFILTER4D:
        {
            shm_set->set_call.set_filter4d = set_filter4d_;
            break;
        }
        case SETACTIVATION:
        {
            shm_set->set_call.set_activation = set_activation_;
            break;
        }
        case SETPOOLING2D:
        {
            shm_set->set_call.set_pooling2d = set_pooling2d_;
            break;
        }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shm_set->sync.barrier); //data prepared
        pthread_barrier_wait(&shm_set->sync.barrier); //tensor set
        pthread_mutex_unlock(&shm_set->sync.mutex);
    }
    ~SetInterProcess()
    {
        int ret;
        ret = munmap(shm_set, SHM_SET_SIZE);
        assert(ret == 0);
    }
};
