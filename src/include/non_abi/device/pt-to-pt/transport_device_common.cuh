/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

 #ifndef _NVSHMEMI_TRANSPORT_DEVICE_COMMON_CUH_
 #define _NVSHMEMI_TRANSPORT_DEVICE_COMMON_CUH_
 
 #include <cuda_runtime.h>
 #include "device/nvshmem_device_macros.h"
 
 #ifdef __CUDA_ARCH__
 
 __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_thread_id_in_warp() {
     int myIdx;
     asm volatile("mov.u32  %0,  %%laneid;" : "=r"(myIdx));
     return myIdx;
 }
 
 __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_warp_size() {
     return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
                ? (blockDim.x * blockDim.y * blockDim.z)
                : warpSize;
 }
 
 __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_warp_sync() { __syncwarp(); }
 
 __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_thread_id_in_block() {
     return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
 }
 
 __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_block_size() {
     return (blockDim.x * blockDim.y * blockDim.z);
 }
 
 #endif /* __CUDA_ARCH__ */
 
 #endif /* _NVSHMEMI_TRANSPORT_DEVICE_COMMON_CUH_ */
 