#include "../ipc.hpp"
class kmeansPoint:public kernel_args
{
	public:
	float  *features,			/* in: [npoints*nfeatures] */
    int     nfeatures;
    int     npoints;
    int     nclusters;
    int    *membership;
	float  *clusters;
	float  *block_clusters;
	int    *block_deltas;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		features=*((float**))args[0]
		nfeatures=*((int*)args[1]);
		npoints=*((int*))args[2];
		nclusters=*((int*)args[3]);
		membership=*((int**)args[4]);
		clusters=*((float**)args[5]);
		block_clusters=*((float*)args[6]);
		block_deltas=*((int**)args[7]);
//		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*5);
        args[0]=(void*)&features;
        args[1]=(void*)&nfeatures;
        args[2]=(void*)&npoints;
		args[3]=(void*)&nclusters;
		args[4]=(void*)&membership;
		args[5]=(void*)&clusters;
		args[6]=(void*)&block_clusters;
		args[7]=(void*)&block_deltas;
		
//	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(kmeansPoint);
	}
}
class kmeansPoint:public kernel_args
{
	public:
	float *input;			/* original */
	float *output;			/* inverted */
	int npoints;				/* npoints */
	int nfeatures;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		input=*((float**))args[0]
		output=*((float**)args[1]);
		npoints=*((int*)args[2]);
		nfeatures=*((int*)args[3]);
//		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*5);
        args[0]=(void*)&input;
        args[1]=(void*)&output;
        args[2]=(void*)&npoints;
		args[3]=(void*)&nfeatures;
		
//	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(invert_mapping);
	}
}