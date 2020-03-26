//#define _GNU_SOURCE
#include "cuda/ipc.hpp"

#include "cstdio"
int new_main(int argc, char **argv, char **nop)
{
    InterProcess ip;
    //   for(int i = 0; i < 100; i++) {
    fprintf(stderr, "libc_wrap: waiting for a kernel\n");
    ip.await_kernel();
    //   }
    return 0;
}

#include "dlfcn.h"
extern "C" int __libc_start_main(__typeof__(new_main) *main, int argc, char **argv,
                                 __typeof__(new_main) *init, void (*fini)(),
                                 void (*rtld_fini)(), void *stack_end)
{
    __typeof__(__libc_start_main) *original =
        (__typeof__(__libc_start_main) *)dlsym(RTLD_NEXT, "__libc_start_main");
    int result = original(new_main, argc, argv,
                          init, fini, rtld_fini, stack_end);
    return result;
}
