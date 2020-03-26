#pragma once
#include </usr/local/cuda/include/cuda_runtime_api.h>
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
#include "check.hpp"

#include <tuple>
#include <vector>

// #define CUDA_CHECK(ans)                          
    // {                                         
        // gpuAssert((ans), __FILE__, __LINE__);
    // }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
// {
    // if (code != cudaSuccess)
    // {
        // fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        // if (abort)
            // exit(code);
    // }
// }

uintptr_t getExcutionStart()
{
    int self_pid = getpid();
    //printf("%d\n", self_pid);
    std::ifstream self_map;
    std::string self_map_file = "/proc/" + std::to_string(self_pid) + "/maps";
    self_map.open(self_map_file, std::ios::in);
    std::string str_line;
    uintptr_t tmp_fptr;
    while (std::getline(self_map, str_line))
    {
        std::stringstream tmp_str(str_line);
        std::string start_end, rwx_flags;
        tmp_str >> start_end;
        tmp_str >> rwx_flags;
        if (rwx_flags == "r-xp")
        {
            std::cout << str_line << std::endl;
            std::stringstream stringto;
            stringto << start_end;
            std::string start;
            std::getline(stringto, start, '-');
            //std::cout << start << std::endl;
            return tmp_fptr = strtol(start.c_str(), NULL, 16);
            break;
        }
    }
    return 1;
}

#define MAX_MEMORY_ALLOCATIONS 16
struct SharedMemoryContents
{
    std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations[MAX_MEMORY_ALLOCATIONS];
    int allocationIndex;
    struct KernelCall
    {
        //const void *fptr;
        uintptr_t fptr;
        dim3 gridDim, blockDim;
        size_t sharedMem;
        int n_args;
        unsigned char arg_buf[4096];
        short arg_offset[128];
        void *argv[128];
    } k_call;
    struct SyncEnforce
    {
        pthread_mutex_t mutex;
        cudaIpcEventHandle_t complete_handle;
        pthread_barrier_t barrier;
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
        pthread_barrier_init(&sync.barrier, &barrierattr, 2);
        pthread_barrierattr_destroy(&barrierattr);
    }
};

#define SHARED_MEM_SIZE sizeof(SharedMemoryContents)
class InterProcess
{
  private:
    SharedMemoryContents *shared_memory;
    const bool is_master;
    cudaStream_t stream;
    cudaEvent_t complete;
    std::vector<void *> pointers_to_close;

  public:
    InterProcess(bool master = false) : is_master(master)
    {
        int fd = shm_open(getenv("SHARED_MEMORY_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
        assert(fd != -1);
        shared_memory = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE,
                                                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        assert(shared_memory != MAP_FAILED);
        assert(close(fd) != -1);
        CUDA_CHECK(cudaStreamCreate(&stream));
        if(!is_master)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming |cudaEventInterprocess));
            CUDA_CHECK(cudaIpcGetEventHandle(&shared_memory->sync.complete_handle, complete));
        }
    }
    void send_kernel(const void *fptr, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem)
    {
        assert(is_master);
        uintptr_t tmp_fptr = getExcutionStart();
        std::cout << tmp_fptr<< std::endl;
        std::cout << fptr<< std::endl;
        //std::cout << tmp_fptr << std::endl;
        //std::cout << (uintptr_t)fptr << std::endl;
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        shared_memory->k_call.fptr = (uintptr_t)fptr - tmp_fptr;
        shared_memory->k_call.gridDim = gridDim;
        shared_memory->k_call.gridDim = blockDim;
        put_args(args);
        shared_memory->k_call.sharedMem = sharedMem;
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
        CUDA_CHECK(cudaIpcOpenEventHandle(&complete, shared_memory->sync.complete_handle));
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel send
        CUDA_CHECK(cudaEventSynchronize(complete));
        // pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
                                                              //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    void await_kernel()
    {
        assert(!is_master);
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        void **args = get_args();
        //std::cout << shared_memory->k_call.fptr << std::endl;
        uintptr_t tmp_fptr = getExcutionStart();
        std::cout << tmp_fptr<< std::endl;
        std::cout << (void *)(shared_memory->k_call.fptr + tmp_fptr) << std::endl;
        CUDA_CHECK(cudaLaunchKernel((void *)(shared_memory->k_call.fptr + tmp_fptr),
                                  shared_memory->k_call.gridDim,
                                  shared_memory->k_call.gridDim, args,
                                  shared_memory->k_call.sharedMem, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(complete));
        release_args();
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
                                                              //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    void add_memory_allocation(void *ptr)
    {
        int idx = shared_memory->allocationIndex++;
        std::get<0>(shared_memory->memoryAllocations[idx]) = ptr;
        cudaIpcGetMemHandle(&(std::get<1>(shared_memory->memoryAllocations[idx])), ptr);
    }
    void *lookup_memory_allocation(void *ptr)
    {
        for (int i = 0; i < shared_memory->allocationIndex; i++)
        {
            if (std::get<0>(shared_memory->memoryAllocations[i]) == ptr)
            {
                void *local_ptr;
                CUDA_CHECK(cudaIpcOpenMemHandle(&local_ptr, std::get<1>(shared_memory->memoryAllocations[i]),
                                             cudaIpcMemLazyEnablePeerAccess));
                return local_ptr;
            }
        }
        return nullptr;
    }
    void put_args(void **args)
    {
        assert(is_master);
        int arg_offset = 0;
        if (args)
        {
            bool last_arg = false;
            int num_args;
            for (num_args = 1; !last_arg && num_args < 128; num_args++)
            {
                size_t diff_prev = pointer_diff(args[num_args - 1], args[num_args]);
                size_t diff_base = pointer_diff(args, args[num_args]);
                if (diff_base > 4096 || diff_prev > 4096)
                    break;
                fprintf(stderr, "arg %d has size %lu\n", num_args - 1, diff_prev);
                memcpy(shared_memory->k_call.arg_buf + arg_offset,
                       args[num_args - 1], diff_prev);
                shared_memory->k_call.arg_offset[num_args - 1] = arg_offset;
                arg_offset += diff_prev;
            }
            assert(arg_offset <= 4096);
            shared_memory->k_call.n_args = num_args;
        }
        else
        {
            shared_memory->k_call.n_args = 0;
        }
    }
    void **get_args()
    {
        assert(!is_master);
        for (int i = 0; i < shared_memory->k_call.n_args; i++)
        {
            int arg_offset = shared_memory->k_call.arg_offset[i];
            void *arg_start = shared_memory->k_call.arg_buf + arg_offset;
            shared_memory->k_call.argv[i] = arg_start;
            void *local_ptr = lookup_memory_allocation(*((void **)arg_start));
            if (local_ptr)
            {
                pointers_to_close.push_back(local_ptr);
                memcpy(arg_start, &local_ptr, sizeof(void *));
                fprintf(stderr, "arg %d is a pointer\n", i);
            }
        }
        //fprintf(stderr, "arg 0: %lu\n", *(size_t*)shared_memory->k_call.argv[0]);
        //fprintf(stderr, "arg 1: %d\n", *(int*)shared_memory->k_call.argv[1]);
        return shared_memory->k_call.argv;
    }
    void release_args()
    {
        for (void *ptr : pointers_to_close)
        {
            CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
        }
        pointers_to_close.clear();
    }
    template <typename T1, typename T2>
    static size_t pointer_diff(T1 one_in, T2 two_in)
    {
        size_t one = (size_t)one_in, two = (size_t)two_in;
        if (one > two)
            return pointer_diff(two, one);
        return two - one;
    }
    ~InterProcess()
    {
        int ret;
        ret = munmap(shared_memory, SHARED_MEM_SIZE);
        assert(ret == 0);
        if (is_master)
        {
            ret = shm_unlink(getenv("SHARED_MEMORY_FNAME"));
            assert(ret == 0);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};
