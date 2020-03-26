
cuda_add_executable(init init.cc)
target_link_libraries(init dl rt pthread glog boost_program_options)
target_compile_options(init PUBLIC -fopenmp)

# cuda_add_library(cudnn_hook SHARED cudnn_hook.cc)
# target_link_libraries(cudnn_hook dl rt)
# target_compile_options(cudnn_hook PUBLIC -fopenmp -shared)

# add_library(optimized_cuda SHARED cuda_hook.cc)
# target_link_libraries(optimized_cuda dl rt)
# target_compile_options(optimized_cuda PUBLIC -fopenmp -shared)

cuda_add_executable(schedule schedule.cc)
target_link_libraries(schedule pthread dl rt glog)

cuda_add_executable(schedule_cublas schedule_cublas.cc)
target_link_libraries(schedule_cublas pthread dl rt glog)
