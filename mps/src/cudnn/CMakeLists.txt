cuda_add_library(tonic_hook SHARED tonic_hook.cc)
target_link_libraries(tonic_hook dl rt glog)
target_compile_options(tonic_hook PUBLIC -fopenmp -shared )

cuda_add_library(cudnn_hook SHARED cudnn_hook.cc)
target_link_libraries(cudnn_hook dl rt glog)
target_compile_options(cudnn_hook PUBLIC -fopenmp -shared )

cuda_add_executable(cudnn_server cudnn_server.cc cudnn_api.cc)
target_link_libraries(cudnn_server cudnn pthread dl rt glog boost_program_options)
