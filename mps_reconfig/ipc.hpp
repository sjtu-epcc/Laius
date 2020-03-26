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
#include <string>
#include <tuple>
#include <vector>
using namespace std;
//#include "kernel_args.h"
//#include "string.h"
 typedef struct {
    string kernel_name;//Kernel's name
    uintptr_t fptr;//kernel's id
    int id;//process id
    int GPU_ratio;//sum of GPU resources
    int process_flag;//-1表示数据没有准备好，0标识数据准备好，你可以调度，调度完成你置为1，2表示schedule端得到信息
 }schedule;
class kernel_args{
	public:
	virtual void from_args(void **args)=0;
	virtual void** to_args()=0;
	virtual int get_size()=0;
};
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
class backprop_k1:public kernel_args
{
	public:
	float* input_cuda;
	float* output_hidden_cuda;
	float* input_hidden_cuda;
	float* hidden_partial_sum;
	int in;
	int hid;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		input_cuda=*((float**)args[0]);
		output_hidden_cuda=*((float**)args[1]);
		input_hidden_cuda=*((float**)args[2]);
		hidden_partial_sum=*((float**)args[3]);
		in=*((int*)args[4]);
		hid=*((int*)args[5]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*6);
        args[0]=(void*)&input_cuda;
        args[1]=(void*)&output_hidden_cuda;
        args[2]=(void*)&input_hidden_cuda;
        args[3]=(void*)&hidden_partial_sum;
        args[4]=(void*)&in;
	args[5]=(void*)&hid;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
}
	int get_size(){
        return sizeof(backprop_k1);
	}
};

class backprop_k2:public kernel_args
{
	public:
	float* delta;
	int hid;
	float* ly;
	int in;
	float* w;
	float* oldw;
	void ** args;
	void from_args(void ** args){
		printf("from_args2\n");
		delta=*((float**)args[0]);
		hid=*((int*)args[1]);
		ly=*((float**)args[2]);
		in=*((int*)args[3]);
		w=*((float**)args[4]);
		oldw=*((float**)args[5]);
		this->args = args;

	}
	void ** to_args(){
	printf("to_args_2\n");
        void ** args=(void **)malloc(sizeof(void *)*6);
        args[0]=(void*)&delta;
        args[1]=(void*)&hid;
        args[2]=(void*)&ly;
        args[3]=(void*)&in;
        args[4]=(void*)&w;
        args[5]=(void*)&oldw;
	args = this->args;
	return args;
	}
	int get_size(){
        return sizeof(backprop_k2);
	}
};



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

#define MAX_MEMORY_ALLOCATIONS 16
#define SHARED_MEM_NUM 30
struct SharedMemoryContents
{
    std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations[MAX_MEMORY_ALLOCATIONS];
    int allocationIndex;
    int order=0;///the order of the kernel that hooked by cudaLaunchkernel
	schedule sch;
    struct KernelCall
    {
        //const void *fptr;
        uintptr_t fptr;
        dim3 gridDim, blockDim;
        size_t sharedMem;
        int n_args;
        unsigned char data[120];
        kernel_args * pargs;
    } k_call;
    struct SyncEnforce
    {
        pthread_mutex_t mutex;
        pthread_barrier_t barrier;
    } sync;
    void init()
    {
		sch.process_flag = -1;
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
        order=0;
        k_call.pargs=(kernel_args*) &k_call.data[0];
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
        cudaStreamCreate(&stream);
    }
    void send_kernel(const void *fptr, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem)
    {
        assert(is_master);
        uintptr_t tmp_fptr = getExcutionStart();
	printf("this is the send_kernel\n");
        //std::cout << tmp_fptr << std::endl;
        //std::cout << (uintptr_t)fptr << std::endl;
       // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        shared_memory->k_call.fptr = (uintptr_t)fptr - tmp_fptr;
        shared_memory->k_call.gridDim = gridDim;
        shared_memory->k_call.blockDim = blockDim;


        ///修改部分
        shared_memory->order++;
        printf("%d\n",shared_memory->order);
       // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
	if(shared_memory->order==1){
            backprop_k1 tmp;
            //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
            memcpy(&shared_memory->k_call.data[0],&tmp,tmp.get_size());
            shared_memory->k_call.pargs=(kernel_args*) &shared_memory->k_call.data[0];
        }else if(shared_memory->order==2){
            backprop_k2 tmp;
            memcpy(&shared_memory->k_call.data[0],&tmp,tmp.get_size());
	   // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
            shared_memory->k_call.pargs=(kernel_args*) &shared_memory->k_call.data[0];
        }
	//printf("First four bytes(int): %d\n",*(int *)shared_memory->k_call.pargs);
        put_args(args);
        shared_memory->k_call.sharedMem = sharedMem;
    	//printf("Sendkernel: args_size: %d\n",shared_memory->k_call.pargs->get_size());
	shared_memory->sch.process_flag = 0;
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    }
    void await_kernel()
    {
		int a;
        assert(!is_master);
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
		printf("this is the await_kernel\n");
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //data ready
	while(shared_memory->sch.process_flag!=1)
	{

//		printf("the process_flag is %d \n",shared_memory->sch.process_flag);
		a = 1;
		usleep(1000);
		a = 1;
	}
	shared_memory->k_call.pargs=(kernel_args*) &shared_memory->k_call.data[0];
      	///Fix
	if(shared_memory->order==1){
		backprop_k1 tmp=*(backprop_k1*)shared_memory->k_call.pargs;
            memcpy(&shared_memory->k_call.data[0],&tmp,tmp.get_size());
	}else if(shared_memory->order==2){
		backprop_k2 tmp=*(backprop_k2*)shared_memory->k_call.pargs;
            memcpy(&shared_memory->k_call.data[0],&tmp,tmp.get_size());
	}
	//printf("%p %p\n",shared_memory->k_call.pargs,&shared_memory->k_call.data[0]); 
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        void **args = get_args();
        //std::cout << shared_memory->k_call.fptr << std::endl;
        uintptr_t tmp_fptr = getExcutionStart();
        cudaLaunchKernel((void *)(shared_memory->k_call.fptr + tmp_fptr),
                                  shared_memory->k_call.gridDim,
                                  shared_memory->k_call.blockDim, args,
                                  shared_memory->k_call.sharedMem, stream);
	printf("await send complete\n");
        cudaStreamSynchronize(stream);
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
	printf("await kernel run the kernel\n");
        release_args();
		shared_memory->sch.process_flag = 2;
		while(shared_memory->sch.process_flag !=-1)
		{
		//| |   |   printf("%d \n",shared_memory->sch.process_flag);|
			a = 1;
			usleep(1000);
			a = 1;
		}
		printf("22222\n");
        pthread_barrier_wait(&(shared_memory->sync.barrier)); //kernel complete
	printf("await kernel finished\n");        
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
       //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
	printf("put_args calls from_args\n");
	shared_memory->k_call.pargs->from_args(args);
        /*
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
        }*/
    }
    void **get_args()
    {
        assert(!is_master);
	printf("get_args call to_args\n");
	//printf("getsize:%d\n",shared_memory->k_call.pargs->get_size());
        return shared_memory->k_call.pargs->to_args();
        /*
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
        return shared_memory->k_call.argv;*/
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
        if (is_master)
        {
            ret = shm_unlink(getenv("SHARED_MEMORY_FNAME"));
            assert(ret == 0);
        }
        cudaStreamDestroy(stream);
    }
};
