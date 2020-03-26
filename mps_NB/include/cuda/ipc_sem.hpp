#pragma once
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include <tuple>
#include <vector>
#include <unordered_map>
#include "cuda/common.hpp"
#include "cuda/kernel_all.hpp"
using namespace std;
struct Schedule
{
    int num;
    uintptr_t id;           //process id
    volatile int GPU_ratio; //sum of GPU resources
    double band;
    volatile bool if_computed;
    volatile double bandwidth;
};

#define MAX_MEMORY_ALLOCATIONS 16
struct SharedMemoryContents
{
    std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations[MAX_MEMORY_ALLOCATIONS];
    volatile int allocationIndex;
    // volatile int order = 0;
    struct KernelCall
    {
        std::uintptr_t fptr;
        dim3 gridDim, blockDim;
        size_t sharedMem;
        unsigned char data[K_SIZE];
    } k_call;
    struct SyncEnforce
    {
        pthread_mutex_t mutex;
        pthread_barrier_t barrier;
        sem_t sch_sem;
        sem_t percent_sem[11];
        sem_t synced;
        sem_t sem_finish;
    } sync;
    Schedule sch;
    void init()
    {
        sch.GPU_ratio = 0;
        sch.id = 0;
        sch.if_computed = false;
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
        sem_init(&sync.sch_sem, 10, 0);
        sem_init(&sync.synced, 10, 0);
        sem_init(&sync.sem_finish, 10, 0);
        for (auto sem_iter : sync.percent_sem)
        {
            sem_init(&sem_iter, 10, 0);
        }
    }
};

#define SHARED_MEM_SIZE sizeof(SharedMemoryContents)
class InterProcess
{
  private:
    SharedMemoryContents *shared_memory;
    const bool is_master;
    cudaStream_t stream;
    int percentage_flag;
    // std::vector<void *> pointers_to_close;
    std::unordered_map<void *, void *> pointers_ready;
    // KernelArgs *kernel_args;
    BfsKernel1 bfs_kernel_1;
    BfsKernel2 bfs_kernel_2;
    BplustreeKernel1 bplustree_kernel_1;
    BplustreeKernel2 bplustree_kernel_2;
    HotspotKernel hotspot_kernel;
    KmeansKernel1 kmeans_kernel_1;
    KmeansKernel2 kmeans_kernel_2;
    LavaMDKernel lavaMD_kernel;
    LudKernel lud_kernel;
    MyocyteKernel1 myocyte_kernel_1;
    MyocyteKernel2 myocyte_kernel_2;
    NwKernel nw_kernel;
    PathfinderKernel pathfinder_kernel;
    void **args;

  public:
    InterProcess(bool master = false) : is_master(master)
    {
        int fd = shm_open(getenv("SHARED_MEMORY_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
        percentage_flag = atoi(getenv("CURRENT_PERCENT"));
        assert(fd != -1);
        shared_memory = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE,
                                                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        assert(shared_memory != MAP_FAILED);
        assert(close(fd) != -1);
        cudachk(cudaStreamCreate(&stream));
    }
    void send_kernel(const void *fptr, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem)
    {
        assert(is_master);

        shared_memory->k_call.fptr = (std::uintptr_t)fptr;
        // std::cout << (std::uintptr_t)fptr << std::endl;
        shared_memory->k_call.gridDim = gridDim;
        shared_memory->k_call.blockDim = blockDim;
        shared_memory->sch.id = shared_memory->k_call.fptr;
        //std::cout << BFS_K1_PTR << std::endl;
        if ((std::uintptr_t)fptr == BFS_K1_PTR)
        {
            bfs_kernel_1.from_args(args);
            memcpy(shared_memory->k_call.data, &bfs_kernel_1, bfs_kernel_1.get_size());
        }
        else if ((std::uintptr_t)fptr == BFS_K2_PTR)
        {
            bfs_kernel_2.from_args(args);
            memcpy(shared_memory->k_call.data, &bfs_kernel_2, bfs_kernel_2.get_size());
        }
        else if ((std::uintptr_t)fptr == BPLUSTREE_K2_PTR)
        {
            fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
            bplustree_kernel_2.from_args(args);
            memcpy(shared_memory->k_call.data, &bplustree_kernel_2, bplustree_kernel_2.get_size());
        }
        else if ((std::uintptr_t)fptr == HOTSOT_K_PTR)
        {
            hotspot_kernel.from_args(args);
            memcpy(shared_memory->k_call.data, &hotspot_kernel, hotspot_kernel.get_size());
        }
        else if ((std::uintptr_t)fptr == KMEANS_K1_PTR)
        {
            kmeans_kernel_1.from_args(args);
            memcpy(shared_memory->k_call.data, &kmeans_kernel_1, kmeans_kernel_1.get_size());
        }
        else if ((std::uintptr_t)fptr == KMEANS_K2_PTR)
        {
            kmeans_kernel_2.from_args(args);
            memcpy(shared_memory->k_call.data, &kmeans_kernel_2, kmeans_kernel_2.get_size());
        }
        else if ((std::uintptr_t)fptr == LAVAMD_K_PTR)
        {
            lavaMD_kernel.from_args(args);
            memcpy(shared_memory->k_call.data, &lavaMD_kernel, lavaMD_kernel.get_size());
        }
        else if ((std::uintptr_t)fptr == LUD_K1_PTR ||
                 (std::uintptr_t)fptr == LUD_K2_PTR ||
                 (std::uintptr_t)fptr == LUD_K3_PTR)
        {
            lud_kernel.from_args(args);
            memcpy(shared_memory->k_call.data, &lud_kernel, lud_kernel.get_size());
        }
        else if ((std::uintptr_t)fptr == MYOCYTE_K1_PTR)
        {
            myocyte_kernel_1.from_args(args);
            memcpy(shared_memory->k_call.data, &myocyte_kernel_1, myocyte_kernel_1.get_size());
        }
        else if ((std::uintptr_t)fptr == MYOCYTE_K2_PTR)
        {
            myocyte_kernel_2.from_args(args);
            memcpy(shared_memory->k_call.data, &myocyte_kernel_2, myocyte_kernel_2.get_size());
        }
        else if ((std::uintptr_t)fptr == NW_K1_PTR ||
                 (std::uintptr_t)fptr == NW_K2_PTR)
        {
            nw_kernel.from_args(args);
            memcpy(shared_memory->k_call.data, &nw_kernel, nw_kernel.get_size());
        }
        else if ((std::uintptr_t)fptr == PATHFINDER_K_PTR)
        {
            pathfinder_kernel.from_args(args);
            memcpy(shared_memory->k_call.data, &pathfinder_kernel, pathfinder_kernel.get_size());
        }
        else
        {
            printf("fptr:%d\n", (std::uintptr_t)fptr);
            printf("kernel parse not supported\n");
            std::abort();
        }
        shared_memory->k_call.sharedMem = sharedMem;
        cudachk(cudaDeviceSynchronize());
        // shared_memory->sch.if_computed = false;
        sem_post(&shared_memory->sync.sch_sem);
        //        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
    }
    void await_kernel()
    {
        assert(!is_master);
        while (true)
        {
            if (sem_trywait(&shared_memory->sync.percent_sem[percentage_flag]) == 0)
            {
                //               printf("this is the %d await Kernel \n",percentage_flag);
                //                pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready

                if (shared_memory->k_call.fptr == BFS_K1_PTR)
                {
                    // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
                    bfs_kernel_1 = *(BfsKernel1 *)shared_memory->k_call.data;
                    args = bfs_kernel_1.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == BFS_K2_PTR)
                {
                    bfs_kernel_2 = *(BfsKernel2 *)shared_memory->k_call.data;
                    args = bfs_kernel_2.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == BPLUSTREE_K2_PTR)
                {
                    fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
                    bplustree_kernel_2 = *(BplustreeKernel2 *)shared_memory->k_call.data;
                    args = bplustree_kernel_2.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == HOTSOT_K_PTR)
                {
                    hotspot_kernel = *(HotspotKernel *)shared_memory->k_call.data;
                    args = hotspot_kernel.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == KMEANS_K1_PTR)
                {
                    kmeans_kernel_1 = *(KmeansKernel1 *)shared_memory->k_call.data;
                    args = kmeans_kernel_1.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == KMEANS_K2_PTR)
                {
                    kmeans_kernel_2 = *(KmeansKernel2 *)shared_memory->k_call.data;
                    args = kmeans_kernel_2.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == LAVAMD_K_PTR)
                {
                    lavaMD_kernel = *(LavaMDKernel *)shared_memory->k_call.data;
                    args = lavaMD_kernel.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == LUD_K1_PTR ||
                         shared_memory->k_call.fptr == LUD_K2_PTR ||
                         shared_memory->k_call.fptr == LUD_K3_PTR)
                {
                    lud_kernel = *(LudKernel *)shared_memory->k_call.data;
                    args = lud_kernel.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == MYOCYTE_K1_PTR)
                {
                    myocyte_kernel_1 = *(MyocyteKernel1 *)shared_memory->k_call.data;
                    args = myocyte_kernel_1.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == MYOCYTE_K2_PTR)
                {
                    myocyte_kernel_2 = *(MyocyteKernel2 *)shared_memory->k_call.data;
                    args = myocyte_kernel_2.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else if (shared_memory->k_call.fptr == NW_K1_PTR ||
                         shared_memory->k_call.fptr == NW_K2_PTR)
                {
                    nw_kernel = *(NwKernel *)shared_memory->k_call.data;
                    args = nw_kernel.to_args(shared_memory->memoryAllocations, pointers_ready, shared_memory->allocationIndex);
                }
                else
                {
                    printf("kernel parse not supported\n");
                    std::abort();
                }

                cudachk(cudaLaunchKernel((void *)(shared_memory->k_call.fptr),
                                         shared_memory->k_call.gridDim,
                                         shared_memory->k_call.blockDim, args,
                                         shared_memory->k_call.sharedMem, stream));
                // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
                free(args);
                shared_memory->sch.if_computed = true;
                cudachk(cudaStreamSynchronize(stream));
                sem_post(&shared_memory->sync.synced);
                //sem_wait(&shared_memory->sync.sem_finish);
                // printf("finished await_kernel\n");
                //                sem_wait(&shared_memory->sync.sem_finish);
                pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
            }
        }
    }
    void add_memory_allocation(void *ptr)
    {
        int idx = shared_memory->allocationIndex++;
        std::get<0>(shared_memory->memoryAllocations[idx]) = ptr;
        cudaIpcGetMemHandle(&(std::get<1>(shared_memory->memoryAllocations[idx])), ptr);
    }
    void release_args()
    {
        for (auto ptr_pair : pointers_ready)
        {
            cudachk(cudaIpcCloseMemHandle(ptr_pair.second));
        }
        pointers_ready.clear();
    }
    bool compute_need()
    {
        return !shared_memory->sch.if_computed;
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
        else
        {
            release_args();
        }
        cudachk(cudaStreamDestroy(stream));
    }
};
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
            //std::cout << str_line << std::endl;
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
