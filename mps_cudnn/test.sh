#!/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

export SHARED_MEMORY_FNAME="cuda_wrapper_ipc"

killall -q origin || true
rm -f "/dev/shm/${SHARED_MEMORY_FNAME}"

echo "origin:"
#./origin

./build/bin/kernelinit
echo
echo "ipc:"
LD_PRELOAD=${DIR}/build/lib/libcuda_wrap.so ./build/bin/origin &
LD_PRELOAD=${DIR}/build/lib/libc_wrap.so ./build/bin/origin &

wait
rm -f "/dev/shm/${SHARED_MEMORY_FNAME}"
