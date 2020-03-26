
cuda_add_executable(optimized_init init.cc)
target_link_libraries(optimized_init dl rt pthread glog boost_program_options)
target_compile_options(optimized_init PUBLIC -fopenmp)

cuda_add_library(optimized_cudnn SHARED cudnn_hook.cc)
target_link_libraries(optimized_cudnn dl rt)
target_compile_options(optimized_cudnn PUBLIC -fopenmp -shared)

# add_library(optimized_cuda SHARED cuda_hook.cc)
# target_link_libraries(optimized_cuda dl rt)
# target_compile_options(optimized_cuda PUBLIC -fopenmp -shared)

cuda_add_executable(optimized_server server.cc cudnn_api.cc)
target_link_libraries(optimized_server cudnn pthread dl rt glog boost_program_options)