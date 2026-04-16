/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef UTILS_INTERNAL
#define UTILS_INTERNAL

#include <stdio.h>
#include <sys/types.h>
#include <inttypes.h>
#include <string>

#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif

/*
 * NVSHMEMTEST_USE_BOOTSTRAP_WITH_MPI = 1 for mpirun launcher (applies to both UID and MPI
 * bootstrap) NVSHMEMTEST_USE_UID_BOOTSTRAP_WITH_HYDRA = 2 for hydra launcher (applies to UID
 * bootstrap only)
 */
enum {
    NVSHMEMTEST_USE_BOOTSTRAP_DEFAULT = 0,
    NVSHMEMTEST_USE_BOOTSTRAP_WITH_MPI,
    NVSHMEMTEST_USE_UID_BOOTSTRAP_WITH_HYDRA,
    NVSHMEMTEST_USE_BOOTSTRAP_UNSUPPORTED,
};

class nvshmemBootstrap {
   public:
    explicit nvshmemBootstrap(int usage) noexcept : usage_mode_(usage) {}
    virtual ~nvshmemBootstrap() noexcept {
        usage_mode_ = static_cast<int>(NVSHMEMTEST_USE_BOOTSTRAP_DEFAULT);
    }
    int usage_mode(void) const { return usage_mode_; }

   private:
    int usage_mode_ = static_cast<int>(NVSHMEMTEST_USE_BOOTSTRAP_DEFAULT);
};

/* ctor/dtor may throw exceptions */
class nvshmemBootstrapMPI final : public nvshmemBootstrap {
   public:
    explicit nvshmemBootstrapMPI(int *c, char ***v);
    ~nvshmemBootstrapMPI() noexcept;
};

/* ctor/dtor may throw exceptions */
class nvshmemBootstrapUID final : public nvshmemBootstrap {
   public:
    explicit nvshmemBootstrapUID(int *c, char ***v);
    ~nvshmemBootstrapUID() noexcept;
};

class nvshmemBootstrapMPIException : public std::exception {
   public:
    nvshmemBootstrapMPIException(std::string &msg) : message(msg) {}
    nvshmemBootstrapMPIException(const char *m) : message(m) {}
    const char *what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

class nvshmemBootstrapUIDException : public std::exception {
   public:
    nvshmemBootstrapUIDException(std::string &msg) : message(msg) {}
    nvshmemBootstrapUIDException(const char *m) : message(m) {}
    const char *what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

#endif /*! UTILS_INTERNAL */
