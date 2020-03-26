
cuda_add_library(cublas_hook SHARED cublas_hook.cc)
target_link_libraries(cublas_hook pthread dl rt glog)
target_compile_options(cublas_hook PUBLIC -fopenmp -shared )

cuda_add_executable(cublas_server cublas_server.cc cublas_api.cc)
target_link_libraries(cublas_server cublas pthread dl rt glog boost_program_options)

cuda_add_executable(cublas_init init.cc)
target_link_libraries(cublas_init dl rt pthread glog boost_program_options)
target_compile_options(cublas_init PUBLIC -fopenmp)

cuda_add_library(cublas_data SHARED cublas_data.cc)
target_link_libraries(cublas_data pthread dl rt glog)
target_compile_options(cublas_data PUBLIC -fopenmp -shared )

cuda_add_executable(test_schedule_cublas schedule.cc)
target_link_libraries(test_schedule_cublas pthread dl rt glog)