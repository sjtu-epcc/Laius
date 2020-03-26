
cuda_add_executable(naive_init init.cc)
target_link_libraries(naive_init dl rt pthread)
target_compile_options(naive_init PUBLIC -fopenmp)

cuda_add_library(naive_wrap SHARED wrap.cc)
target_link_libraries(naive_wrap dl rt)
target_compile_options(naive_wrap PUBLIC -fopenmp -shared)

cuda_add_executable(naive_server server.cc)
target_link_libraries(naive_server cudnn pthread dl rt glog boost_program_options)
