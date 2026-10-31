/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_EXAMPLES_BOOTSTRAP_HELPER_
#define _NVSHMEMI_EXAMPLES_BOOTSTRAP_HELPER_

#ifdef NVSHMEMTEST_MPI_SUPPORT
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "mpi.h"

typedef int (*fnptr_MPI_Init)(int *argc, char ***argv);
typedef int (*fnptr_MPI_Bcast)(void *buffer, int count, MPI_Datatype datatype, int root,
                               MPI_Comm comm);
typedef int (*fnptr_MPI_Comm_rank)(MPI_Comm comm, int *rank);
typedef int (*fnptr_MPI_Comm_size)(MPI_Comm comm, int *size);
typedef int (*fnptr_MPI_Finalize)(void);
struct nvshmemi_mpi_fn_table {
    fnptr_MPI_Init fn_MPI_Init;
    fnptr_MPI_Bcast fn_MPI_Bcast;
    fnptr_MPI_Comm_rank fn_MPI_Comm_rank;
    fnptr_MPI_Comm_size fn_MPI_Comm_size;
    fnptr_MPI_Finalize fn_MPI_Finalize;
};

void *nvshmemi_mpi_handle = NULL;
struct nvshmemi_mpi_fn_table mpi_fn_table = {0};
MPI_Comm MPI_COMM_WORLD_PLACEHOLDER;
MPI_Datatype MPI_UINT8_T_PLACEHOLDER;
MPI_Datatype *mpi_uint8_ptr;

#define MPI_LOAD_SYM(fn_name)                                                          \
    mpi_fn_table.fn_##fn_name = (fnptr_##fn_name)dlsym(nvshmemi_mpi_handle, #fn_name); \
    if (mpi_fn_table.fn_##fn_name == NULL) {                                           \
        fprintf(stderr, "Unable to load MPI symbol" #fn_name "\n");                    \
        return -1;                                                                     \
    }

/* Open MPI's predefined handles are addresses of objects inside libmpi, so they are looked up
 * in the dlopen'ed library. MPICH-derived MPIs define them as integer constants in mpi.h, which
 * are usable without linking libmpi. NVSHMEMTEST_MPI_LIB overrides the library name. */
#ifdef OPEN_MPI
#define NVSHMEMTEST_MPI_LIB_DEFAULT "libmpi.so.40"
#define NVSHMEMTEST_MPI_HANDLE(type, handle, ompi_symbol) \
    ((type)dlsym(nvshmemi_mpi_handle, ompi_symbol))
#else
#define NVSHMEMTEST_MPI_LIB_DEFAULT "libmpi.so.12"
#define NVSHMEMTEST_MPI_HANDLE(type, handle, ompi_symbol) (handle)
#endif

int nvshmemi_load_mpi() {
    const char *mpi_lib = getenv("NVSHMEMTEST_MPI_LIB");
    if (mpi_lib == NULL) {
        mpi_lib = NVSHMEMTEST_MPI_LIB_DEFAULT;
    }
    nvshmemi_mpi_handle = dlopen(mpi_lib, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    if (nvshmemi_mpi_handle == NULL) {
        // Print the error number and description from errno.
        fprintf(stderr, "dlopen failed: errno = %d, description = %s\n", errno, strerror(errno));

        // Additionally, print the error message from dlerror for more specific information.
        const char *dlerror_msg = dlerror();
        if (dlerror_msg) {
            fprintf(stderr, "dlerror: %s\n", dlerror_msg);
        }
        fprintf(stderr,
                "Unable to dlopen %s. "
                "Please add it to your LD_LIBRARY_PATH, set NVSHMEMTEST_MPI_LIB or run without"
                " NVSHMEMTEST_USE_MPI_LAUNCHER.\n",
                mpi_lib);
        return -1;
    }
    MPI_LOAD_SYM(MPI_Init);
    MPI_LOAD_SYM(MPI_Bcast);
    MPI_LOAD_SYM(MPI_Comm_rank);
    MPI_LOAD_SYM(MPI_Comm_size);
    MPI_LOAD_SYM(MPI_Finalize);

    return 0;
}

void nvshmemi_init_mpi(int *c, char ***v) {
    int status;
    int rank, nranks;

    status = nvshmemi_load_mpi();
    if (status) {
        exit(-1);
    }

    mpi_fn_table.fn_MPI_Init(c, v);

    MPI_COMM_WORLD_PLACEHOLDER =
        NVSHMEMTEST_MPI_HANDLE(MPI_Comm, MPI_COMM_WORLD, "ompi_mpi_comm_world");
    MPI_UINT8_T_PLACEHOLDER = NVSHMEMTEST_MPI_HANDLE(MPI_Datatype, MPI_UINT8_T, "ompi_mpi_uint8_t");

    mpi_fn_table.fn_MPI_Comm_rank(MPI_COMM_WORLD_PLACEHOLDER, &rank);
    mpi_fn_table.fn_MPI_Comm_size(MPI_COMM_WORLD_PLACEHOLDER, &nranks);

    MPI_Comm mpi_comm = MPI_COMM_WORLD_PLACEHOLDER;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
}

int nvshmemi_dlclose_mpi() {
    int status;

    status = dlclose(nvshmemi_mpi_handle);
    if (status) {
        fprintf(stderr, "unable to dlclose MPI.\n");
        return -1;
    }
    return 0;
}

void nvshmemi_finalize_mpi() {
    mpi_fn_table.fn_MPI_Finalize();
    nvshmemi_dlclose_mpi();
}

#endif  // NVSHMEMTEST_MPI_SUPPORT
#endif  // _NVSHMEMI_EXAMPLES_BOOTSTRAP_HELPER_
