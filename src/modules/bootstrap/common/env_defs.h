/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 */

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

#ifndef NVSHMEM_ENV_DEFS_INTERNAL
#include "bootstrap_host_transport/env_defs_internal.h"  // IWYU pragma: keep
#endif

#ifdef NVSHMEMI_ENV_DEF

NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Bootstrap **/
NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_SOCK_IFNAME, string, "", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the UID bootstrap socket interface name")

NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_SOCK_FAMILY, string, "AF_INET", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the UID bootstrap socket family name")

NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_SESSION_ID, string, "", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the UID bootstrap session identifier")

NVSHMEMI_ENV_DEF(BOOTSTRAP_SHMEM_MODE, string, "auto", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Select the OpenSHMEM bootstrap collective path.\n"
                 "Allowed values: auto, legacy, teams")

NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_SEND_DRAIN_MS, int, 1000, NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Timeout in milliseconds for draining send queue before closing connection.\n"
                 "Used to ensure proper connection ordering under RSS/softirq-induced reordering.\n"
                 "Default: 1000ms, <=0 disables the feature")

/** Debugging **/
NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "Values: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
#endif
