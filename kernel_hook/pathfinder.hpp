#include "../ipc.hpp"
class dynproc_kernel:public kernel_args
{
	public:
	int iteration;
    int *gpuWall;
    int *gpuSrc;
    int *gpuResults;
    int cols; 
    int rows;
    int startStep;
    int borde;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		iteration=*((int*)args[0]);
		gpuWall=*((int**)args[1]);
		gpuSrc=*((int**)args[2]);
		gpuResults=*((int**)args[3]);
		cols=*((int*)args[4]);
		rows=*((int*)args[5]);
		startStep=*((int*)args[6]);
		borde=*((int*)args[7]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*7);
        args[0]=(void*)&iteration;
        args[1]=(void*)&gpuWall;
        args[2]=(void*)&gpuSrc;
		args[3]=(void*)&gpuResults;
        args[4]=(void*)&cols;
        args[5]=(void*)&rows;
		args[6]=(void*)&startStep;
		args[7]=(void*)&borde;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(dynproc_kernel);
	}
}