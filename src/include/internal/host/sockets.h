/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SOCKETS_H
#define SOCKETS_H

#include <sys/types.h>

typedef struct ipcHandle_st {
    int socket;
    char *socketName;
} ipcHandle;

int ipcOpenSocket(ipcHandle *&handle, pid_t, pid_t);

int ipcCloseSocket(ipcHandle *handle);

int ipcRecvFd(ipcHandle *handle, int *fd);

int ipcSendFd(ipcHandle *handle, const int fd, pid_t process, pid_t);
int ipcCloseFd(int fd);

#endif /* SOCKETS_H */
