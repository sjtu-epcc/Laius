#include "../ipc.hpp"
class hotspot:public kernel_args
{
	public:
	int iteration;  //number of iteration
    float *power;   //power input
    float *temp_src;    //temperature input/output
    float *temp_dst;    //temperature input/output
    int grid_cols;  //Col of grid
    int grid_rows;  //Row of gri
	int border_cols;  // border offset 
	int border_rows; // border offset
    float Cap;      //Capacitance
    float Rx; 
    float Ry; 
    float Rz; 
    float step; 
    float time_elapsed;
	void **args;
	void from_args(void ** args){
		printf("from_args1\n");
		iteration=*((int*))args[0]
		power=*((float**)args[1]);
		temp_src=*((float**)args[2]);
		temp_dst=*((float**)args[3]);
		grid_cols=*((int*)args[4]);
		grid_rows=*((int*)args[5]);
		border_cols=*((int*)args[6]);
		border_rows=*((int*)args[7]);
		Cap=*((float*)args[8]);
		Rx=*((float*)args[9]);
		Ry=*((float*)args[10]);
		Rz=*((float*)args[11]);
		step=*((float*)args[12]);
		time_elapsed=*((float*)args[13]);
//		printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//		this->args = args;
	}
	void ** to_args(){
	//fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
        printf("to_args1\n");
	void ** args=(void **)malloc(sizeof(void *)*5);
        args[0]=(void*)&iteration;
        args[1]=(void*)&power;
        args[2]=(void*)&temp_src;
		args[3]=(void*)&temp_dst;
		args[4]=(void*)&grid_cols;
		args[5]=(void*)&grid_rows;
		args[6]=(void*)&border_cols;
		args[7]=(void*)&border_rows;
		args[8]=(void*)&Cap;
		args[9]=(void*)&Rx;
		args[10]=(void*)&Ry;
		args[11]=(void*)&Rz;
		args[12]=(void*)&step;
		args[13]=(void*)&time_elapsed;
		
//	printf("%d %d %d %d %d %d\n",input_cuda,output_hidden_cuda,input_hidden_cuda,hidden_partial_sum,in,hid);
//	args = this->args;	
	return args;
	int get_size(){
        return sizeof(hotspot);
	}
}