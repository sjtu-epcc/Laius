#pragma once
#include "cuda/apps/kernel_args.hpp"

// #define fp float
typedef struct par_str
{
    float alpha;
} par_str;

typedef struct dim_str
{

    // input arguments
    int cur_arg;
    int arch_arg;
    int boxes1d_arg;

    // system memory
    long number_boxes;
    long box_mem;
    long space_elem;
    long space_mem;
    long space_mem2;

} dim_str;
class LavaMDKernel : public KernelArgs
{
  private:
    par_str d_par_gpu;
    dim_str d_dim_gpu;
    void *d_box_gpu;
    void *d_rv_gpu;
    void *d_qv_gpu;
    void *d_fv_gpu;

  public:
    LavaMDKernel() : KernelArgs(LAVAMD_K_PTR) {}
    void from_args(void **args_)
    {
        d_par_gpu = *(par_str *)args_[0];
        d_dim_gpu = *(dim_str *)args_[1];
        d_box_gpu = *(void **)args_[2];
        d_rv_gpu = *(void **)args_[3];
        d_qv_gpu = *(void **)args_[4];
        d_fv_gpu = *(void **)args_[5];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 6);
        args[0] = (void *)&d_par_gpu;
        args[1] = (void *)&d_dim_gpu;
        d_box_gpu = lookup_memory_allocation(d_box_gpu, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &d_box_gpu;
        d_rv_gpu = lookup_memory_allocation(d_rv_gpu, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &d_rv_gpu;
        d_qv_gpu = lookup_memory_allocation(d_qv_gpu, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &d_qv_gpu;
        d_fv_gpu = lookup_memory_allocation(d_fv_gpu, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &d_fv_gpu;
        return args;
    }
    int get_size()
    {
        return sizeof(LavaMDKernel);
    }
};