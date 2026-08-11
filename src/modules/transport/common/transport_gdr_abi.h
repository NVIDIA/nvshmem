/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_GDR_ABI_H
#define _TRANSPORT_GDR_ABI_H

#include <stddef.h>
#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>

/*
 * Keep the small, stable portion of the GDRCopy ABI local. The transport
 * loads libgdrapi dynamically, so CUDA DMA-BUF support does not require
 * gdrapi.h at build time.
 */
struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
    unsigned long h;
} gdr_mh_t;

struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped : 1;
    unsigned wc_mapping : 1;
    /* Present in GDRCopy 2.6's gdr_info_v2. Older libraries leave it unused. */
    int mapping_type;
};
typedef struct gdr_info gdr_info_t;

static constexpr int NVSHMEMT_GDR_API_MAJOR_VERSION = 2;

/* GDRCopy values used only with symbols resolved through dlsym. */
static constexpr uint32_t NVSHMEMT_GDR_PIN_FLAG_FORCE_PCIE = 1U;
static constexpr int NVSHMEMT_GDR_MAP_FLAG_DEFAULT = 0;
static constexpr int NVSHMEMT_GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2;
static constexpr int NVSHMEMT_GDR_ATTR_USING_DMA_BUF_MMAP = 3;

#endif
