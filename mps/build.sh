#!/bin/bash -e
###
#File: /home/haohao/Projects/Paper/reference/cudnn/build.sh
#Project: /home/haohao/Projects/Paper/reference/cudnn
#Created Date: Monday, December 24th 2018, 5:07:30 pm
#Author: Raphael-Hao
#-----
#Last Modified: Monday, January 14th 2019, 9:52:13 pm
#Modified By: Raphael-Hao
#-----
#Copyright (c) 2018 Happy
#
#Were It to Benefit My Country, I Would Lay Down My Life !
###

function usage() {
	echo "Usage  : ./build.sh [clean/simple/clean]"
	echo "total  : clean all old built files and rebuild the whole project"
	echo "simple : just rebuild the project"
	echo "clean  : just clean shared memory"
}

function remove_shm() {
	if [ -e /dev/shm/shared_compute ]; then
		rm /dev/shm/shared_compute
	fi
	if [ -e /dev/shm/shared_set ]; then
		rm /dev/shm/shared_set
	fi
	if [ -e /dev/shm/shared_malloc ]; then
		rm /dev/shm/shared_malloc
	fi
}

if [ "$1" == "total" ]; then
	git pull && rm -rf build && mkdir build &&
		cd build &&
		rm -rf * &&
		cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF .. &&
		make -"j$(nproc)" &&
		cd ..
	remove_shm
	echo " Total rebuilding succeed "
elif [ "$1" == "simple" ]; then
	git pull &&
		cd build &&
		cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF .. &&
		make -"j$(nproc)" &&
		cd ..
	remove_shm
	echo " Simple rebuilding succeed "
elif [ "$1" == "clean" ]; then
	remove_shm
	echo "All created shared memory removed"
else
	usage
fi
