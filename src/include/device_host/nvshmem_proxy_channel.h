/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEM_PROXY_CHANNEL_H_
#define _NVSHMEM_PROXY_CHANNEL_H_

#if !defined __CUDACC_RTC__
#include <stdint.h>
#else
#include <cuda/std/cstdint>
#endif

/* Note: this is only safe because we are using this across a single system
 * that shares the GPUs endianness. This struct is not actually portable.
 */
typedef union channel_bounce_buffer {
    char bytes[8];
    uint64_t whole_buffer;
} channel_bounce_buffer_t;

/* base_request_t
 * 32 | 8 | 8 | 8 | 8
 * roffset_high | roffset_low | op | group_size | flag */
typedef struct __attribute__((packed)) base_request {
    volatile uint8_t flag;
    uint8_t groupsize;
    uint8_t op;
    uint8_t roffset_low;   // target is remote
    uint32_t roffset_high; /*used as pe for base-only requests*/
} base_request_t;
static_assert(sizeof(base_request_t) == 8, "request_size must be 8 bytes.");

/* put_dma_request_0
 * 32 | 16 | 8 | 8
 * laddr_high | laddr_3| laddr_2 | flag */
typedef struct __attribute__((packed)) put_dma_request_0 {
    volatile uint8_t flag;
    uint8_t laddr_2;
    uint16_t laddr_3;  // source is local
    uint32_t laddr_high;
} put_dma_request_0_t;
static_assert(sizeof(put_dma_request_0) == 8, "request_size must be 8 bytes.");

/* put_dma_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | laddr_low | flag */
typedef struct __attribute__((packed)) put_dma_request_1 {
    volatile uint8_t flag;
    uint8_t laddr_low;
    uint16_t size_low;
    uint32_t size_high;
} put_dma_request_1_t;
static_assert(sizeof(put_dma_request_1) == 8, "request_size must be 8 bytes.");

/* put_dma_request_2
 * 32 | 16 | 8 | 8
 * resv2 | pe | resv | flag */
typedef struct __attribute__((packed)) put_dma_request_2 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t resv1;
} put_dma_request_2_t;
static_assert(sizeof(put_dma_request_2) == 8, "request_size must be 8 bytes.");

/* put_inline_request_0
 * 32 | 16 | 8 | 8
 * loffset_high | loffset_low | pe | flag */
typedef struct __attribute__((packed)) put_inline_request_0 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t lvalue_low;
} put_inline_request_0_t;
static_assert(sizeof(put_inline_request_0) == 8, "request_size must be 8 bytes.");

/* put_inline_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | resv | flag */
typedef struct __attribute__((packed)) put_inline_request_1 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t size;
    uint32_t lvalue_high;
} put_inline_request_1_t;
static_assert(sizeof(put_inline_request_1) == 8, "request_size must be 8 bytes.");

/* amo_request_0
 * 32 | 16 | 8 | 8
 * lvalue_low | pe | amo | flag */
typedef struct __attribute__((packed)) amo_request_0 {
    volatile uint8_t flag;
    uint8_t amo;
    uint16_t pe;
    uint32_t swap_add_low;
} amo_request_0_t;
static_assert(sizeof(amo_request_0) == 8, "request_size must be 8 bytes.");

/* amo_request_1
 * 32 | 16 | 8 | 8
 * lvalue_high | resv | size | flag */
typedef struct __attribute__((packed)) amo_request_1 {
    volatile uint8_t flag;
    uint8_t compare_low;
    uint16_t size;
    uint32_t swap_add_high;
} amo_request_1_t;
static_assert(sizeof(amo_request_1) == 8, "request_size must be 8 bytes.");

/* amo_request_2
 * 56 | 8
 * compare_high | flag */
typedef struct __attribute__((packed)) amo_request_2 {
    volatile uint8_t flag;
    uint8_t compare_high[7];
} amo_request_2_t;
static_assert(sizeof(amo_request_2) == 8, "request_size must be 8 bytes.");

typedef struct __attribute__((packed)) amo_request_3 {
    volatile uint8_t flag;
    uint8_t g_buf_counter[7];
} amo_request_3_t;
static_assert(sizeof(amo_request_3) == 8, "request_size must be 8 bytes.");

/*
 * PUT_SIGNAL REQUEST STRUCTURE DEFINITIONS
 *
 * The put_signal operation requires 48 bytes total (6 x 8-byte entries) to encode:
 * - Base Request for identifying type of operation and target offset
 * - RMA parameters
 * - Atomic parameters
 *
 * Each 8-byte entry follows the pattern: [data_bits][flag_byte]
 * Flag alternates 0/1 based on circular buffer iteration for lock-free synchronization.
 *
 * MEMORY LAYOUT DIAGRAM:
 *
 * Entry 0 (base_request_t):
 * [rwrite_offset_high:32][rwrite_offset_low:8][op:8][group_size:8][flag:8] Entry 1 (put_signal_0):
 * [laddr_write_high:32][laddr_write_3:16][laddr_write_2:8][flag:8] Entry 2 (put_signal_1):
 * [write_size_high:32][write_size_low:16][laddr_write_low:8][flag:8] Entry 3 (put_signal_2):
 * [reserved:32][pe:16][reserved:8][flag:8] Entry 4 (put_signal_3):
 * [rsigoffset_high:32][rsigoffset_low:8][sig_op:8][sigval_low:8][flag:8] Entry 5 (put_signal_4):
 * [sigval_high:32][sigval_3:16][sigval_2:8][flag:8]
 *
 */

/* put_signal_request_0
 * 56               | 8
 * laddr_write_high | flag */
typedef struct __attribute__((packed)) put_signal_request_0 {
    volatile uint8_t flag;
    uint8_t laddr_write_2;
    uint16_t laddr_write_3;
    uint32_t laddr_write_high;
} put_signal_request_0_t;
static_assert(sizeof(put_signal_request_0) == 8, "request_size must be 8 bytes.");

/* put_signal_request_1
 * 32              | 16             | 8               | 8
 * write_size_high | write_size_low | lwrite_addr_low | flag */
typedef struct __attribute__((packed)) put_signal_request_1 {
    volatile uint8_t flag;
    uint8_t laddr_write_low;
    uint16_t write_size_low;
    uint32_t write_size_high;
} put_signal_request_1_t;
static_assert(sizeof(put_signal_request_1) == 8, "request_size must be 8 bytes.");

/* put_signal_request_2
 * 32    | 16 | 8     | 8
 * resv2 | pe | resv1 | flag */
typedef struct __attribute__((packed)) put_signal_request_2 {
    volatile uint8_t flag;
    uint8_t resv1;
    uint16_t pe;
    uint32_t resv2;
} put_signal_request_2_t;
static_assert(sizeof(put_signal_request_2) == 8, "request_size must be 8 bytes.");

/* put_signal_request_3
 * 32              | 8              | 8      | 8          | 8
 * rsigoffset_high | rsigoffset_low | sig_op | sigval_low | flag */
typedef struct __attribute__((packed)) put_signal_request_3 {
    volatile uint8_t flag;
    uint8_t sigval_low;
    uint8_t sig_op;
    uint8_t rsigoffset_low;
    uint32_t rsigoffset_high;
} put_signal_request_3_t;
static_assert(sizeof(put_signal_request_3) == 8, "request_size must be 8 bytes.");

/* put_signal_request_4
 * 56          | 8
 * sigval_high | flag */
typedef struct __attribute__((packed)) put_signal_request_4 {
    volatile uint8_t flag;
    uint8_t sigval_2;
    uint16_t sigval_3;
    uint32_t sigval_high;
} put_signal_request_4_t;
static_assert(sizeof(put_signal_request_4) == 8, "request_size must be 8 bytes.");

#endif
