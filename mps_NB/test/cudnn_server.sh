#!/bin/bash -e
###
#File: /home/haohao/Projects/Paper/reference/cudnn/test_optimized.sh
#Project: /home/haohao/Projects/Paper/reference/cudnn
#Created Date: Tuesday, December 18th 2018, 2:36:16 pm
#Author: Raphael-Hao
#-----
#Last Modified: Wednesday, January 16th 2019, 2:24:45 pm
#Modified By: Raphael-Hao
#-----
#Copyright (c) 2018 Happy
#
#Were It to Benefit My Country, I Would Lay Down My Life !
###

if [ -e /dev/shm/shared_compute ]; then
	rm /dev/shm/shared_compute
fi
if [ -e /dev/shm/shared_set ]; then
	rm /dev/shm/shared_set
fi
if [ -e /dev/shm/cuda_0 ]; then
	rm /dev/shm/cuda_0
fi
if [ -e /dev/shm/cuda_1 ]; then
	rm /dev/shm/cuda_1
fi
if [ -e /dev/shm/cuda_2 ]; then
	rm /dev/shm/cuda_2
fi
if [ -e /dev/shm/cuda_3 ]; then
	rm /dev/shm/cuda_3
fi
if [ -e /dev/shm/cuda_4 ]; then
	rm /dev/shm/cuda_4
fi
if [ -e /dev/shm/cuda_5 ]; then
	rm /dev/shm/cuda_5
fi
if [ -e /dev/shm/cuda_6 ]; then
	rm /dev/shm/cuda_6
fi
if [ -e /dev/shm/cuda_7 ]; then
	rm /dev/shm/cuda_7
fi
if [ -e /dev/shm/cuda_8 ]; then
	rm /dev/shm/cuda_8
fi
if [ -e /dev/shm/cuda_9 ]; then
	rm /dev/shm/cuda_9
fi
if [ -e /dev/shm/cuda_10 ]; then
	rm /dev/shm/cuda_10
fi
if [ -e /dev/shm/cuda_11 ]; then
	rm /dev/shm/cuda_11
fi
if [ -e /dev/shm/cuda_12 ]; then
	rm /dev/shm/cuda_12
fi
if [ -e /dev/shm/cuda_13 ]; then
	rm /dev/shm/cuda_13
fi
if [ -e /dev/shm/cuda_14 ]; then
	rm /dev/shm/cuda_14
fi
if [ -e /dev/shm/cuda_15 ]; then
	rm /dev/shm/cuda_15
fi
if [ -e /dev/shm/cuda_16 ]; then
	rm /dev/shm/cuda_16
fi
if [ -e /dev/shm/cuda_17 ]; then
	rm /dev/shm/cuda_17
fi
if [ -e /dev/shm/cuda_18 ]; then
	rm /dev/shm/cuda_18
fi
if [ -e /dev/shm/cuda_19 ]; then
	rm /dev/shm/cuda_19
fi
#killall  cudnn_server
export PREDICT_DATA_PATH=/home/whcui/Laius/mps/data
export SHARED_COMPUTE_FNAME=shared_compute
export SHARED_SET_FNAME=shared_set
#export CUDA_VISIBLE_DEVICES=2

./build/bin/init -p 6

#proce=`echo get_server_list |nvidia-cuda-mps-control`
#echo set_active_thread_percentage $proce 10|nvidia-cuda-mps-control
#./build/bin/cudnn_server -p 10 &
#proce=`echo get_server_list |nvidia-cuda-mps-control`
#echo set_active_thread_percentage $proce 20|nvidia-cuda-mps-control
#./build/bin/cudnn_server -p 20 &
#proce=`echo get_server_list |nvidia-cuda-mps-control`
#echo set_active_thread_percentage $proce 30|nvidia-cuda-mps-control
#./build/bin/cudnn_server -p 30 &
# proce=`echo get_server_list |nvidia-cuda-mps-control`
# echo set_active_thread_percentage $proce 40|nvidia-cuda-mps-control
#./build/bin/cudnn_server -p 40 &
#proce=`echo get_server_list |nvidia-cuda-mps-control`
#echo set_active_thread_percentage $proce 50|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 50 &
 proce=`echo get_server_list |nvidia-cuda-mps-control`
 echo set_active_thread_percentage $proce 60|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 60 &
 proce=`echo get_server_list |nvidia-cuda-mps-control`
 echo set_active_thread_percentage $proce 70|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 70 &
 proce=`echo get_server_list |nvidia-cuda-mps-control`
 echo set_active_thread_percentage $proce 80|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 80 &
 proce=`echo get_server_list |nvidia-cuda-mps-control`
 echo set_active_thread_percentage $proce 90|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 90 &
 proce=`echo get_server_list |nvidia-cuda-mps-control`
 echo set_active_thread_percentage $proce 100|nvidia-cuda-mps-control
./build/bin/cudnn_server -p 100 
wait
