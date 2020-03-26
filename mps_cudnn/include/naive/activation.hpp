#include "naive/base_api.hpp"

class activation : base_api
{
  public:
    cudnnActivationDescriptor_t tensor_act;
    activation()
    {
        cudnnCreateActivationDescriptor(&tensor_act);
    }
    void setup(SharedMemoryContents *shared_memory)
    {
        base_setup(shared_memory);
        CUDNN_CHECK(cudnnSetActivationDescriptor(tensor_act, shared_memory->cudnn_call.activation_desc.mode,
                                                 shared_memory->cudnn_call.activation_desc.reluNanOpt,
                                                 shared_memory->cudnn_call.activation_desc.coef));
    }
    void compute(cudnnHandle_t &cudnn_handle,cudaEvent_t &event_, SharedMemoryContents *shared_memroy)
    {
        CUDNN_CHECK(cudnnActivationForward(cudnn_handle, tensor_act, &alpha,
                                           tensor_in, tensor_in_ptr, &beta,
                                           tensor_out, tensor_out_ptr));
        // if (shared_memroy->change_flag == 1)
        // {
            // synchronize(shared_memroy);
        // }
        shared_memroy->process_flag = -1;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_in_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_out_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaEventRecord(event_));
        // CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shared_memroy->sync.barrier);
    }
};