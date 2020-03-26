#include "../ipc.hpp"
typedef struct record {
	int value;
} record;
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 
class b+tree:public kernel_args{
public:
	long height;
	knode *knodesD;
	long knodes_elem;
	record *recordsD;
	long *currKnodeD;
	long *offsetD;
	int *keysD;
	record *ansD;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		height=*((long*)args[0]);
		knodesD=*((knode**)args[1]);
		knodes_elem=*((long*)args[2]);
		recordsD=*((record**)args[3]);
		currKnodeD=*((long**)args[4]);
		offsetD=*((long**)args[5]);
		keysD=*((int**)args[6]);
		ansD=*((record**)args[7]);
		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*6);
        args[0]=(void*)&height;
        args[1]=(void*)&knodesD;
        args[2]=(void*)&knodes_elem;
        args[3]=(void*)&recordsD;
        args[4]=(void*)&currKnodeD;
		args[5]=(void*)&offsetD;
		args[6]=(void*)&keysD;
		args[7]=(void*)&ansD;
	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;}
	int get_size(){
        return sizeof(b+tree);
	}
};