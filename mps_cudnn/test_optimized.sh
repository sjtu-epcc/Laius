#!/bin/bash -e
###
#File: /home/haohao/Projects/Paper/reference/mps/test_optimized.sh
#Project: /home/haohao/Projects/Paper/reference/mps
#Created Date: Tuesday, December 18th 2018, 2:36:16 pm
#Author: Raphael-Hao
#-----
#Last Modified: Tuesday, December 25th 2018, 1:57:38 pm
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
if [ -e /dev/shm/shared_malloc ]; then
	rm /dev/shm/shared_malloc
fi
SHARED_MALLOC_FNAME=shared_malloc SHARED_COMPUTE_FNAME=shared_compute SHARED_SET_FNAME=shared_set ./build/bin/optimized_init

SHARED_MALLOC_FNAME=shared_malloc SHARED_COMPUTE_FNAME=shared_compute SHARED_SET_FNAME=shared_set ./build/bin/optimized_server
