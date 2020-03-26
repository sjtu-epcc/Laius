#pragma once
#include "cuda/apps/kernel_args.hpp"

class BfsKernel1 : public KernelArgs
{
  private:
    void *g_graph_nodes;
    void *g_graph_edges;
    void *g_graph_mask;
    void *g_updating_graph_mask;
    void *g_graph_visited;
    void *g_cost;
    int no_of_nodes;

  public:
    BfsKernel1() : KernelArgs(BFS_K1_PTR) {}
    void from_args(void **args_)
    {
        // printf("from_args1\n");
        // void **args_ = (void **)malloc(sizeof(void *) * 7);
        g_graph_nodes = *(void **)args_[0];
        g_graph_edges = *(void **)args_[1];
        g_graph_mask = *(void **)args_[2];
        g_updating_graph_mask = *(void **)args_[3];
        g_graph_visited = *(void **)args_[4];
        g_cost = *(void **)args_[5];
        no_of_nodes = *((int *)args_[6]);
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        // printf("to_args1\n");
        void **args = (void **)malloc(sizeof(void *) * 7);
        // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        g_graph_nodes = lookup_memory_allocation(g_graph_nodes, memoryAllocations_, pointers_ready_, memIndex_);
        // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        args[0] = &g_graph_nodes;
        g_graph_edges = lookup_memory_allocation(g_graph_edges, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &g_graph_edges;
        g_graph_mask = lookup_memory_allocation(g_graph_mask, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &g_graph_mask;
        g_updating_graph_mask = lookup_memory_allocation(g_updating_graph_mask, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &g_updating_graph_mask;
        g_graph_visited = lookup_memory_allocation(g_graph_visited, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &g_graph_visited;
        g_cost = lookup_memory_allocation(g_cost, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &g_cost;
        args[6] = (void *)&no_of_nodes;
        return args;
    }
    int get_size()
    {
        return sizeof(BfsKernel1);
    }
};

class BfsKernel2 : public KernelArgs
{
  private:
    void *g_graph_mask;
    void *g_updating_graph_mask;
    void *g_graph_visited;
    void *g_over;
    int no_of_nodes;

  public:
    BfsKernel2() : KernelArgs(BFS_K2_PTR) {}
    void from_args(void **args_)
    {
        // printf("from_args2\n");
        g_graph_mask = *(void **)args_[0];
        g_updating_graph_mask = *(void **)args_[1];
        g_graph_visited = *(void **)args_[2];
        g_over = *(void **)args_[3];
        no_of_nodes = *(int *)args_[4];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        //fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        // printf("to_args2\n");
        void **args = (void **)malloc(sizeof(void *) * 5);
        g_graph_mask = lookup_memory_allocation(g_graph_mask, memoryAllocations_, pointers_ready_, memIndex_);
        args[0] = &g_graph_mask;
        g_updating_graph_mask = lookup_memory_allocation(g_updating_graph_mask, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &g_updating_graph_mask;
        g_graph_visited = lookup_memory_allocation(g_graph_visited, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &g_graph_visited;
        g_over = lookup_memory_allocation(g_over, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &g_over;
        args[4] = (void *)&no_of_nodes;
        return args;
    }
    int get_size()
    {
        return sizeof(BfsKernel2);
    }
};
