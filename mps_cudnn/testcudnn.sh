#!/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

export SHARED_MEMORY_FNAME="cudnn_wrapper_ipc"

if [ $1 = 1 ]; then
	git pull && cd build && make -j8 && cd ..
fi
if [ $1 = 2 ]; then
	git pull && cd build && cmake .. && make clean && make -j8 && cd ..
fi

killall -q cudnn_server || true
rm -f "/dev/shm/${SHARED_MEMORY_FNAME}"

echo "origin:"
#./origin

./build/bin/cudnninit
echo
echo "ipc:"
./build/bin/cudnn_server 0

wait
rm -f "/dev/shm/${SHARED_MEMORY_FNAME}"
