#pragma once
#include "cuda/apps/kernel_args.hpp"

class HotspotKernel : public KernelArgs
{
  private:
    int iteration;
    void *power;
    void *temp_src;
    void *temp_dst;
    int grid_cols;
    int grid_rows;
    int border_cols; // border offset
    int border_rows; // border offset
    float Cap;       // Capacitance
    float Rx, Ry, Rz, step;

  public:
    HotspotKernel() : KernelArgs(HOTSOT_K_PTR) {}
    void from_args(void **args_)
    {
        iteration = *(int *)args_[0];
        power = *(void **)args_[1];
        temp_src = *(void **)args_[2];
        temp_dst = *(void **)args_[3];
        grid_cols = *(int *)args_[4];
        grid_rows = *(int *)args_[5];
        border_cols = *(int *)args_[6];
        border_rows = *(int *)args_[7];
        Cap = *(float *)args_[8];
        Rx = *(float *)args_[9];
        Ry = *(float *)args_[10];
        Rz = *(float *)args_[11];
        step = *(float *)args_[12];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 13);
        args[0] = (void *)&iteration;
        power = lookup_memory_allocation(power, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &power;
        temp_src = lookup_memory_allocation(temp_src, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &temp_src;
        temp_dst = lookup_memory_allocation(temp_dst, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &temp_dst;
        args[4] = (void *)&grid_cols;
        args[5] = (void *)&grid_rows;
        args[6] = (void *)&border_cols;
        args[7] = (void *)&border_rows;
        args[8] = (void *)&Cap;
        args[9] = (void *)&Rx;
        args[10] = (void *)&Ry;
        args[11] = (void *)&Rz;
        args[12] = (void *)&step;
        return args;
    }
    int get_size()
    {
        return sizeof(HotspotKernel);
    }
};
