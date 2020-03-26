#include <boost/program_options.hpp>
#include "naive/addtensor.hpp"
#include "naive/convolution.hpp"
#include "naive/activation.hpp"
#include "naive/pooling.hpp"
#include "naive/softmax.hpp"
#include "naive/cudnn_ipc.hpp"

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
        "percent,p", boost::program_options::value<int>(&cur_percent)->implicit_value(10), "percentage of mps resource usage");
    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opts), var_map);
    boost::program_options::notify(var_map);
    LOG(INFO) << "Server started, Using " << cur_percent << "sources!";
    // map shared memory for compute
    int fd = shm_open(getenv("SHARED_MEMORY_FNAME"), O_RDWR, S_IRUSR | S_IWUSR);
    void *init_ptr = mmap(NULL, SHARED_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    uintptr_t init_addr = (uintptr_t)init_ptr;
    SharedMemoryContents *shared_memory[SHARED_MEM_NUM];
    for (int i = 0; i < SHARED_MEM_NUM; i++)
    {
        shared_memory[i] = (SharedMemoryContents *)(init_addr + SHM_FLAG_SIZE + i * SHARED_MEM_SIZE);
    }
    LOG(INFO) << "Reading Shared Memory Succeed!";
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    cudaEvent_t complete;
    cudaEvent_t start;
    CUDA_CHECK(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming | cudaEventInterprocess));

    addtensor addtensor_solver;
    convolution convolution_solver;
    activation activation_solver;
    pooling pooling_solver;
    softmax softmax_solver;
    LOG(INFO) << "Wait for Cudnn API!";
    while (true)
    {
        for (int i = 0; i < SHARED_MEM_NUM; i++)
        {
            // LOG(INFO) << i << "-th shared memory block "<< "percent: " << shared_memory[i]->percent_flag<< " pid: " << shared_memory[i]->pid_flag<< " process: " << shared_memory[i]->process_flag<< std::endl;
            if (shared_memory[i]->percent_flag == cur_percent &&
                shared_memory[i]->pid_flag == -1 &&
                shared_memory[i]->process_flag == 0)
            {
                pthread_mutex_lock(&shared_memory[i]->sync.mutex);
                if (shared_memory[i]->pid_flag == -1)
                {
                    shared_memory[i]->pid_flag = cur_pid;
                    pthread_mutex_unlock(&shared_memory[i]->sync.mutex);
                    CUDA_CHECK(cudaIpcGetEventHandle(&shared_memory[i]->sync.complete_handle, complete));
                    CUDA_CHECK(cudaIpcOpenEventHandle(&start, shared_memory[i]->sync.start_handle));
                    pthread_barrier_wait(&shared_memory[i]->sync.barrier);
                    // LOG(INFO) << shared_memory[i]->cudnn_call.api_type<< std::endl;
                    switch (shared_memory[i]->cudnn_call.api_type)
                    {
                    case ADDTENSOR_N:
                        LOG(INFO) << "ADDTENSOR received" << std::endl;
                        addtensor_solver.setup(shared_memory[i]);
                        CUDA_CHECK(cudaEventSynchronize(start));
                        addtensor_solver.compute(cudnn, complete, shared_memory[i]);
                        break;
                    case CONVFWD_N:
                        LOG(INFO) << "CONVFWD received" << std::endl;
                        convolution_solver.setup(shared_memory[i]);
                        CUDA_CHECK(cudaEventSynchronize(start));
                        convolution_solver.compute(cudnn, complete, shared_memory[i]);
                        break;
                    case ACTIVATIONFWD_N:
                        LOG(INFO) << "ACTIVATIONFWD received" << std::endl;
                        activation_solver.setup(shared_memory[i]);
                        CUDA_CHECK(cudaEventSynchronize(start));
                        activation_solver.compute(cudnn, complete, shared_memory[i]);
                        break;
                    case POOLFWD_N:
                        LOG(INFO) << "POOLFWD received" << std::endl;
                        pooling_solver.setup(shared_memory[i]);
                        CUDA_CHECK(cudaEventSynchronize(start));
                        pooling_solver.compute(cudnn, complete, shared_memory[i]);
                        break;
                    case SOFTMAXFWD_N:
                        LOG(INFO) << "SOFTMAXFWD received" << std::endl;
                        softmax_solver.setup(shared_memory[i]);
                        CUDA_CHECK(cudaEventSynchronize(start));
                        softmax_solver.compute(cudnn, complete, shared_memory[i]);
                        break;
                        // default:
                        // LOG(INFO) << "No such cudnn api" << std::endl;
                        // exit(0);
                    }
                }
                else
                    continue;
            }
            else
                continue;
        }
    }
    return 0;
}