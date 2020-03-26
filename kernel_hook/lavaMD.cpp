#include "../ipc.h"
typedef struct
{
	fp x, y, z;

} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;

} FOUR_VECTOR;

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;

typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;

typedef struct par_str
{

	fp alpha;

} par_str;

typedef struct dim_str
{

	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;

} dim_str;
class kernel_gpu_cuda:public kernel_args
{
	public:
	par_str d_par_gpu;
	dim_str d_dim_gpu;
	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		d_par_gpu=*((par_str*))args[0]
		d_dim_gpu=*((dim_str*)args[1]);
		d_box_gpu=*((box_str**))args[2];
		d_rv_gpu=*((FOUR_VECTOR**)args[3]);
		d_qv_gpu=*((fp**)args[4]);
		d_fv_gpu=*((FOUR_VECTOR**)args[5]);
//		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*5);
        args[0]=(void*)&d_par_gpu;
        args[1]=(void*)&d_dim_gpu;
        args[2]=(void*)&d_box_gpu;
		args[3]=(void*)&d_rv_gpu;
		args[4]=(void*)&d_qv_gpu;
		args[5]=(void*)&d_fv_gpu;
		
//	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(kernel_gpu_cuda);
	}
}