
add_library(cuda_hook SHARED cuda_hook.cc)
target_link_libraries(cuda_hook dl rt)
target_compile_options(cuda_hook PUBLIC -fopenmp -shared)

add_library(c_hook SHARED c_hook.cc)
target_link_libraries(c_hook dl rt)
target_compile_options(c_hook PUBLIC -fopenmp -shared)

add_library(ptr_hook SHARED ptr_hook.cc)
target_link_libraries(ptr_hook dl rt)
target_compile_options(ptr_hook PUBLIC -fopenmp -shared)

# cuda_add_executable(cudnn_server server.cc cudnn_api.cc)
# target_link_libraries(cudnn_server cudnn pthread dl rt glog boost_program_options)