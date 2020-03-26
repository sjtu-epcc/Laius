#pragma once
#include "cuda/apps/kernel_args.hpp"

class KmeansKernel1 : public KernelArgs
{
  private:
    void *input;
    void *output;
    int n_points;
    int n_features;

  public:
    KmeansKernel1() : KernelArgs(KMEANS_K1_PTR) {}
    void from_args(void **args_)
    {
        input = *(void **)args_[0];
        output = *(void **)args_[1];
        n_points = *(int *)args_[2];
        n_features = *(int *)args_[3];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 4);
        input = lookup_memory_allocation(input, memoryAllocations_, pointers_ready_, memIndex_);
        args[0] = &input;
        output = lookup_memory_allocation(output, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &output;
        args[2] = (void *)&n_points;
        args[3] = (void *)&n_features;
        return args;
    }
    int get_size()
    {
        return sizeof(KmeansKernel1);
    }
};

class KmeansKernel2 : public KernelArgs
{
  private:
    void *features;
    int n_features;
    int n_points;
    int n_clusters;
    void *membership;
    void *clusters;
    void *block_clusters;
    void *block_deltas;

  public:
    KmeansKernel2() : KernelArgs(KMEANS_K2_PTR) {}
    void from_args(void **args_)
    {
        features = *(void **)args_[0];
        n_features = *(int *)args_[1];
        n_points = *(int *)args_[2];
        n_clusters = *(int *)args_[3];
        membership = *(void **)args_[4];
        clusters = *(void **)args_[5];
        block_clusters = *(void **)args_[6];
        block_deltas = *(void **)args_[7];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 8);
        features = lookup_memory_allocation(features, memoryAllocations_, pointers_ready_, memIndex_);
        args[0] = &features;
        args[1] = (void *)&n_features;
        args[2] = (void *)&n_points;
        args[3] = (void *)&n_clusters;
        membership = lookup_memory_allocation(membership, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &membership;
        clusters = lookup_memory_allocation(clusters, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &clusters;
        block_clusters = lookup_memory_allocation(block_clusters, memoryAllocations_, pointers_ready_, memIndex_);
        args[6] = &block_clusters;
        block_deltas = lookup_memory_allocation(block_clusters, memoryAllocations_, pointers_ready_, memIndex_);
        args[7] = &block_deltas;
        return args;
    }
    int get_size()
    {
        return sizeof(KmeansKernel2);
    }
};