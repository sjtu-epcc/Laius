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
#killall  cudnn_server
export PREDICT_DATA_PATH=/home/laius/Laius/mps/data/
export SHARED_COMPUTE_FNAME=shared_compute
export SHARED_SET_FNAME=shared_set
export QOS_DATA=face.txt
#export QOS_DATA=googlenet.txt
#export QOS_DATA=vgg.txt
./build/bin/cuda_schedule
