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
#include <cudnn.h>
#include "check.hpp"

// #include <vector>

#define SHARED_MEM_NUM 20
struct alignas(16) ShmUseFlag
{
    int used[SHARED_MEM_NUM];
    // int launched[20];
    pthread_mutex_t mutex;
    void init()
    {
        for (int i = 0; i < SHARED_MEM_NUM; i++)
        {
            used[i] = 0;
        }
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&mutex, &mutexattr);
    }
};
#define SHM_FLAG_SIZE sizeof(ShmUseFlag)
struct alignas(16) TensorDesc
{
    // uintptr_t device_ptr;
    cudaIpcMemHandle_t device_handle;
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
};

struct alignas(16) ConvDesc
{
    // uintptr_t device_ptr;
    int pad_h, pad_w, u, v, dilation_h, dilation_w;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t computeType;
    cudaIpcMemHandle_t workspace_handle;
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t algo;
};
struct alignas(16) FilterDesc
{
    // uintptr_t device_ptr;
    cudaIpcMemHandle_t device_handle;
    cudnnTensorFormat_t filter_format;
    cudnnDataType_t dataType;
    int k, c, h, w;
};

struct alignas(16) ActivationDesc
{
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t reluNanOpt;
    double coef;
};

struct alignas(16) PoolingDesc
{
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    int windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride;
};

struct alignas(16) SoftmaxInfo
{
    cudnnSoftmaxAlgorithm_t algo;
    cudnnSoftmaxMode_t mode;
};

#define MAX_MEMORY_ALLOCATIONS 16
struct alignas(16) SharedMemoryContents
{
    int percent_flag; //标志用多少百分比的资源计算
    int pid_flag;     //用来让进程池的进程来知道是不是自己来计算这个api，这个你不用管
    int process_flag; //0标识数据准备好，你可以调度，调度完成你置为1
    // int change_flag;//-1标识不用调度，0标识不用切换，1标识需要切换
    struct alignas(16) cudnnAPI
    {
        API_TYPE api_type;
        float alpha, beta;
        TensorDesc tensor_in;
        TensorDesc tensor_out;
        TensorDesc tensor_etc;
        ConvDesc conv_desc;
        FilterDesc filter_desc;
        ActivationDesc activation_desc;
        PoolingDesc pooling_desc;
        SoftmaxInfo softmax_info;
    } cudnn_call;
    struct alignas(16) SyncEnforce
    {
        pthread_mutex_t mutex;
        // pthread_barrier_t data_barrier;
        pthread_barrier_t barrier;
        cudaIpcEventHandle_t complete_handle;
        cudaIpcEventHandle_t start_handle;
    } sync;
    int init()
    {
        pid_flag = -1;
        percent_flag = 0;
        process_flag = -1;
        // change_flag = -1;
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&sync.mutex, &mutexattr);
        pthread_mutexattr_destroy(&mutexattr);
        pthread_barrierattr_t barrierattr;
        pthread_barrierattr_init(&barrierattr);
        pthread_barrierattr_setpshared(&barrierattr, PTHREAD_PROCESS_SHARED);
        // pthread_barrier_init(&sync.data_barrier, &barrierattr, 2);    //todo: nums of barriers
        pthread_barrier_init(&sync.barrier, &barrierattr, 2); //todo: nums of barriers
        pthread_barrierattr_destroy(&barrierattr);
        return 1;
    }
};

#define SHARED_MEM_SIZE sizeof(SharedMemoryContents)
#define SHARED_SIZE SHARED_MEM_NUM *SHARED_MEM_SIZE + SHM_FLAG_SIZE
class InterProcess
{
  private:
    SharedMemoryContents *shared_memory;
    ShmUseFlag *shm_use_flag;
    int using_flag;
    const bool is_master;
    cudaEvent_t complete;
    cudaEvent_t start;
    // cudaStream_t stream;
    // std::vector<void *> pointers_to_close;

  public:
    InterProcess(bool master = false) : is_master(master)
    {
        int fd = shm_open(getenv("SHARED_MEMORY_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
        assert(fd != -1);
        void *init_ptr = mmap(NULL, SHARED_SIZE,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, 0);
        uintptr_t init_addr = (uintptr_t)init_ptr;
        assert(get_free_shm(init_addr) != -1);
        assert(close(fd) != -1);
        if (is_master)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming | cudaEventInterprocess));
            CUDA_CHECK(cudaIpcGetEventHandle(&shared_memory->sync.start_handle, start));
        }
        // cudachk(cudaStreamCreate(&stream));
    }
    void send_cudnn(API_TYPE &api_type_, float &alpha_, float &beta_,
                    TensorDesc &tensor_in_, TensorDesc &tensor_out_, TensorDesc &tensor_etc_,
                    ConvDesc &conv_desc_, FilterDesc &filter_desc_, ActivationDesc &activation_desc_,
                    PoolingDesc &pooling_desc_, SoftmaxInfo &sofmax_info_)
    {
        assert(is_master);
        while (shared_memory->process_flag != -1)
            ;
        shared_memory->cudnn_call.api_type = api_type_;
        shared_memory->cudnn_call.alpha = alpha_;
        shared_memory->cudnn_call.beta = beta_;
        shared_memory->cudnn_call.tensor_in = tensor_in_;
        shared_memory->cudnn_call.tensor_out = tensor_out_;
        shared_memory->cudnn_call.tensor_etc = tensor_etc_;
        shared_memory->cudnn_call.conv_desc = conv_desc_;
        shared_memory->cudnn_call.filter_desc = filter_desc_;
        shared_memory->cudnn_call.activation_desc = activation_desc_;
        shared_memory->cudnn_call.pooling_desc = pooling_desc_;
        shared_memory->cudnn_call.softmax_info = sofmax_info_;
        shared_memory->process_flag = 0;
        shared_memory->pid_flag = -1;
        // shared_memory->
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //event recorded
        CUDA_CHECK(cudaIpcOpenEventHandle(&complete, shared_memory->sync.complete_handle));
        CUDA_CHECK(cudaEventRecord(start));
        pthread_barrier_wait(&(shared_memory->sync.barrier));
        CUDA_CHECK(cudaEventSynchronize(complete)); //kernel complete
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    int get_free_shm(uintptr_t init_addr_)
    {
        shm_use_flag = (ShmUseFlag *)init_addr_;
        assert(shm_use_flag != MAP_FAILED);
        pthread_mutex_lock(&shm_use_flag->mutex);
        for (int i = 0; i < SHARED_MEM_NUM; i++)
        {
            if (shm_use_flag->used[i] == 0)
            {
                shm_use_flag->used[i] = 1;
                // shm_use_flag->launched[i] = 0;
                pthread_mutex_unlock(&shm_use_flag->mutex);
                using_flag = i;
                shared_memory = (SharedMemoryContents *)(init_addr_ + SHM_FLAG_SIZE + i * SHARED_MEM_SIZE);
                return i;
            }
        }
        return -1;
    }
    // void release_args()
    // {
    //     for (void *ptr : pointers_to_close)
    //     {
    //         CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    //     }
    //     pointers_to_close.clear();
    // }
    ~InterProcess()
    {
        memset(shared_memory, 0, SHARED_MEM_SIZE);
        shared_memory->init();
        int ret;
        ret = munmap(shm_use_flag, SHARED_SIZE);
        assert(ret == 0);
        // if (is_master)
        // {
        // ret = shm_unlink(getenv("SHARED_MEMORY_FNAME"));
        // assert(ret == 0);
        // }
        // cudachk(cudaStreamDestroy(stream));
    }
};