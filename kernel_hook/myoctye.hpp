#include "../ipc.hpp"
class lud_diagonal:public kernel_args
{
	public:
	int timeinst;
	fp* d_initvalu;
	fp* d_finavalu;
	fp* d_params;
	fp* d_com;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		m=*((float**)args[0]);
		matrix_dim=*((int*)args[1]);
		offset=*((int*)args[2]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*7);
        args[0]=(void*)&m;
        args[1]=(void*)&matrix_dim;
        args[2]=(void*)&offset;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(lud_diagonal);
	}
}
