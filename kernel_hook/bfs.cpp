#include "../ipc.hpp"
struct Node
{
	int starting;
	int no_of_edges;
};
class Kernel:public kernel_args
{
	public:
	Node* g_graph_nodes;
	int* g_graph_edges;
	bool* g_graph_mask;
	bool* g_updating_graph_mask;
	bool *g_graph_visited;
	int* g_cost;
	int no_of_nodes;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		g_graph_nodes=*((Node**)args[0]);
		g_graph_edges=*((int**)args[1]);
		g_graph_mask=*((bool**)args[2]);
		g_updating_graph_mask=*((bool**)args[3]);
		g_graph_visited=*((bool**)args[4]);
		g_cost=*((int**)args[5]);
		no_of_nodes=*((int*)args[6]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*7);
        args[0]=(void*)&g_graph_nodes;
        args[1]=(void*)&g_graph_edges;
        args[2]=(void*)&g_graph_mask;
        args[3]=(void*)&g_updating_graph_mask;
        args[4]=(void*)&g_graph_visited;
		args[5]=(void*)&g_cost;
		args[6]=(void*)&no_of_nodes
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(Kernel);
	}
}
class Kernel2:public kernel_args
{
	public:
	bool* g_graph_mask;
	bool *g_updating_graph_mask;
	bool* g_graph_visited;
	bool *g_over;
	int no_of_nodes;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		g_graph_mask=*((bool**)args[0]);
		g_updating_graph_mask=*((bool**)args[1]);
		g_graph_visited=*((bool**)args[2]);
		g_over=*((bool**)args[3]);
		g_cost=*((int**)args[4]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*5);
        args[0]=(void*)&g_graph_mask;
        args[1]=(void*)&g_updating_graph_mask;
        args[2]=(void*)&g_graph_visited;
		args[3]=(void*)&g_over;
		args[4]=(void*)&g_cost;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(Kernel2);
	}
}