cuda_add_library(origin_wrap SHARED wrap.cc)
target_link_libraries(origin_wrap dl rt)
target_compile_options(origin_wrap PUBLIC -fopenmp -shared)