cuda_add_executable(kernel_init init.cc)
target_link_libraries(kernel_init dl rt pthread)
target_compile_options(kernel_init PUBLIC -fopenmp)

add_library(kernel_cuda_wrap SHARED cuda_wrap.cc)
target_link_libraries(kernel_cuda_wrap dl rt)

add_library(kernel_c_wrap SHARED c_wrap.cc)
target_link_libraries(kernel_c_wrap dl rt glog)
target_compile_options(kernel_c_wrap PUBLIC -fopenmp -shared)

cuda_add_executable(origin origin.cu)
# target_link_libraries(origin cudart)
# add_executable(testIpc testIpc.cc)
# target_link_libraries(testIpc dl rt pthread)