#include "naive/base_api.hpp"

class softmax : base_api
{
  public:
    // cudnnSoftmaxAlgorithm_t algo;
    // cudnnSoftmaxMode_t mode;
    // softmax()
    // {
    // }
    void setup(SharedMemoryContents *shared_memory)
    {
        base_setup(shared_memory);
    }
    void compute(cudnnHandle_t &cudnn_handle,cudaEvent_t &event_, SharedMemoryContents *shared_memory)
    {
        CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handle,
                                        shared_memory->cudnn_call.softmax_info.algo,
                                        shared_memory->cudnn_call.softmax_info.mode,
                                        &alpha, tensor_in, tensor_in_ptr,
                                        &beta, tensor_out, tensor_out_ptr));
        // if(shared_memory->change_flag == 1)
        // {
            // synchronize(shared_memory);
        // }
        shared_memory->process_flag = -1;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_in_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_out_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaEventRecord(event_));
        // CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shared_memory->sync.barrier);
    }
};