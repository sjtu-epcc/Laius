#!/bin/bash -e

export SHARED_COMPUTE_FNAME=shared_compute
bin/init
echo init program has finished...
#killall -q origin || true
#rm -f "/dev/shm/${SHARED_MEMORY_FNAME}"
#./schedule
bin/schedule
wait
bin/uninit
