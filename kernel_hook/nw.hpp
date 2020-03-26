#include "../ipc.hpp"
class needle_cuda_shared_1:public kernel_args
{
	public:
	int* referrence;
	int* matrix_cuda; 
	int cols;
	int penalty;
	int i;
	int block_widt;
	void from_args(void ** args){
		printf("from_args1\n");
		referrence=*((int**)args[0]);
		matrix_cuda=*((int**)args[1]);
		penalty=*((int*)args[2]);
		cols=*((int*)args[3]);
		i=*((int*)args[4]);
		block_widt=*((int*)args[5]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*7);
        args[0]=(void*)&referrence;
        args[1]=(void*)&matrix_cuda;
        args[2]=(void*)&penalty;
		args[3]=(void*)&cols;
		args[4]=(void*)&i;
		args[5]=(void*)&block_widt;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(needle_cuda_shared_1);
	}
}
class needle_cuda_shared_2:public kernel_args
{
	public:
	int* referrence;
	int* matrix_cuda; 
	int cols;
	int penalty;
	int i;
	int block_widt;
	void from_args(void ** args){
		printf("from_args1\n");
		referrence=*((int**)args[0]);
		matrix_cuda=*((int**)args[1]);
		penalty=*((int*)args[2]);
		cols=*((int*)args[3]);
		i=*((int*)args[4]);
		block_widt=*((int*)args[5]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*7);
        args[0]=(void*)&referrence;
        args[1]=(void*)&matrix_cuda;
        args[2]=(void*)&penalty;
		args[3]=(void*)&cols;
		args[4]=(void*)&i;
		args[5]=(void*)&block_widt;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(needle_cuda_shared_2);
	}
}