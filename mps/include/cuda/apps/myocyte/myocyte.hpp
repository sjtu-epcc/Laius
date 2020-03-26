#pragma once

#include "cuda/apps/kernel_args.hpp"

class MyocyteKernel1 : public KernelArgs
{
  private:
    int timeinst;
    void *d_initvalu;
    void *d_finavalu;
    void *d_params;
    void *d_com;

  public:
    MyocyteKernel1() : KernelArgs(MYOCYTE_K1_PTR) {}
    void from_args(void **args_)
    {
        timeinst = *(int *)args_[0];
        d_initvalu = *(void **)args_[1];
        d_finavalu = *(void **)args_[2];
        d_params = *(void **)args_[3];
        d_com = *(void **)args_[4];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 5);
        args[0] = (void *)&timeinst;
        d_initvalu = lookup_memory_allocation(d_initvalu, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &d_initvalu;
        d_finavalu = lookup_memory_allocation(d_finavalu, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &d_finavalu;
        d_params = lookup_memory_allocation(d_params, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &d_params;
        d_com = lookup_memory_allocation(d_com, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &d_com;
        return args;
    }
    int get_size()
    {
        return sizeof(MyocyteKernel1);
    }
};

class MyocyteKernel2 : public KernelArgs
{
  private:
    int workload;
    int xmax;
    void *x;
    void *y;
    void *params;
    void *com;
    void *err;
    void *scale;
    void *yy;
    void *initvalu_temp;
    void *finavalu_temp;

  public:
    MyocyteKernel2() : KernelArgs(MYOCYTE_K2_PTR) {}
    void from_args(void **args_)
    {
        workload = *(int *)args_[0];
        xmax = *(int *)args_[1];
        x = *(void **)args_[2];
        y = *(void **)args_[3];
        params = *(void **)args_[4];
        com = *(void **)args_[5];
        err = *(void **)args_[6];
        scale = *(void **)args_[7];
        yy = *(void **)args_[8];
        initvalu_temp = *(void **)args_[9];
        finavalu_temp = *(void **)args_[10];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 11);
        args[0] = (void *)&workload;
        args[1] = (void *)&xmax;
        x = lookup_memory_allocation(x, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &x;
        y = lookup_memory_allocation(y, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &y;
        params = lookup_memory_allocation(params, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &params;
        com = lookup_memory_allocation(com, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &com;
        err = lookup_memory_allocation(err, memoryAllocations_, pointers_ready_, memIndex_);
        args[6] = &err;
        scale = lookup_memory_allocation(scale, memoryAllocations_, pointers_ready_, memIndex_);
        args[7] = &scale;
        yy = lookup_memory_allocation(yy, memoryAllocations_, pointers_ready_, memIndex_);
        args[8] = &yy;
        initvalu_temp = lookup_memory_allocation(initvalu_temp, memoryAllocations_, pointers_ready_, memIndex_);
        args[9] = &initvalu_temp;
        finavalu_temp = lookup_memory_allocation(finavalu_temp, memoryAllocations_, pointers_ready_, memIndex_);
        args[10] = &finavalu_temp;
        return args;
    }
    int get_size()
    {
        return sizeof(MyocyteKernel2);
    }
};