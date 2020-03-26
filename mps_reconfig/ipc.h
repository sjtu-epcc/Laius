#pragma once
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include <tuple>
#include <vector>
using namespace std;
#define cudachk(ans)                          \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
/*struct alignas(16) ShmUseFlag
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
};*/
vector<string> getProcessName()
{
	vector<string> process_list;
	process_list.push_back("cuda_0");
	process_list.push_back("cuda_1");
	process_list.push_back("cuda_2");
	process_list.push_back("cuda_3");
	process_list.push_back("cuda_4");
	process_list.push_back("cuda_5");
	process_list.push_back("cuda_6");
	process_list.push_back("cuda_7");
	process_list.push_back("cuda_8");
	process_list.push_back("cuda_9");
	process_list.push_back("cuda_10");
	process_list.push_back("cuda_11");
	process_list.push_back("cuda_12");
	process_list.push_back("cuda_13");
	process_list.push_back("cuda_14");
	process_list.push_back("cuda_15");
	process_list.push_back("cuda_16");
	process_list.push_back("cuda_17");
	process_list.push_back("cuda_18");
	process_list.push_back("cuda_19");
	return process_list;
}
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
typedef struct {
	string kernel_name;//Kernel's name
	uintptr_t fptr;//kernel's id
	int id;//process id
	int GPU_ratio;//sum of GPU resources
	int process_flag;//-1表示数据没有准备好，0标识数据准备好，你可以调度，调度完成你置为1，2表示schedule端得到信息
}schedule;
#define MAX_MEMORY_ALLOCATIONS 16
#define SHARED_MEM_NUM 30
//#define SHM_FLAG_SIZE sizeof(ShmUseFlag)
#define SHARED_MEM_SIZE sizeof(SharedMemoryContents)
#define SHARED_SIZE SHARED_MEM_NUM *SHARED_MEM_SIZE
struct SharedMemoryContents
{
    std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations[MAX_MEMORY_ALLOCATIONS];
    int allocationIndex;
    schedule sch;//the info of kernel's schedule
    int ifsync;
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
        pthread_barrier_init(&sync.barrier, &barrierattr, 11);
        pthread_barrierattr_destroy(&barrierattr);
	sch.process_flag=-1;
	sch.GPU_ratio=0;
		
    }
};

#define SHARED_MEM_SIZE sizeof(SharedMemoryContents)
class InterProcess
{
  private:
    SharedMemoryContents *shared_memory;
    const bool is_master;
    cudaStream_t stream;
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
	shared_memory->init();
        cudachk(cudaStreamCreate(&stream));
    }
    void send_kernel(const void *fptr, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem)
    {
        assert(is_master);
        uintptr_t tmp_fptr = getExcutionStart();
        //std::cout << tmp_fptr << std::endl;
        //std::cout << (uintptr_t)fptr << std::endl;
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        shared_memory->k_call.fptr = (uintptr_t)fptr - tmp_fptr;
        shared_memory->k_call.gridDim = gridDim;
        shared_memory->k_call.gridDim = blockDim;
	printf("the fptr is %d\n",tmp_fptr);
        put_args(args);
        shared_memory->k_call.sharedMem = sharedMem;
	//shared_memory->sch.change_flag = 1;
	shared_memory->sch.process_flag = 0;
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
                                                              //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    void await_kernel()
    {
	int a;
	//char str[40];
        assert(!is_master);
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
	while(shared_memory->sch.process_flag!=1)
	{
//		printf("%d \n",shared_memory->sch.process_flag);
		a = 1;
		usleep(1000);
		a = 1;
	}
//	printf("11111\n");
//	printf("22222\n");
//	printf("%s\n",str);
	//system(str);
	//FILE *f;
//	f=popen(str,"r");
//	pclose(f);
//	printf("33333\n");
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
//	printf("this is the %d th process\n",atoi(getenv("GPU_PERCENTAGE")));
	if(shared_memory->sch.GPU_ratio == atoi(getenv("GPU_PERCENTAGE")))
 	{
		printf("the Kernel is %d\n percentage",10*shared_memory->sch.GPU_ratio);
		void **args = get_args();
        	//std::cout << shared_memory->k_call.fptr << std::endl;
        	uintptr_t tmp_fptr = getExcutionStart();
        	cudachk(cudaLaunchKernel((void *)(shared_memory->k_call.fptr + tmp_fptr),
                                  shared_memory->k_call.gridDim,
                                  shared_memory->k_call.gridDim, args,
                                  shared_memory->k_call.sharedMem, stream));
        	cudachk(cudaStreamSynchronize(stream));
        	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        	release_args();
		//shared_memory->sch.process_flag = -1;
//		shared_memory->sch.change_flag = -1;
        	shared_memory->sch.process_flag = 2;
      		while(shared_memory->sch.process_flag !=-1)
		{
//			printf("%d \n",shared_memory->sch.process_flag);	
			a = 1;
			usleep(1000);
			a = 1;
		}
	}
	pthread_barrier_wait(&(shared_memory->sync.barrier));	
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
                cudachk(cudaIpcOpenMemHandle(&local_ptr, std::get<1>(shared_memory->memoryAllocations[i]),
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
            cudachk(cudaIpcCloseMemHandle(ptr));
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
/*        if (is_master)
        {
            ret = shm_unlink(getenv("SHARED_MEMORY_FNAME"));
            assert(ret == 0);
        }*/
        cudachk(cudaStreamDestroy(stream));
    }
};
