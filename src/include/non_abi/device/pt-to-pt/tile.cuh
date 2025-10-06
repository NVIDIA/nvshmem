/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TILE_PT_TO_PT_CUH__
#define __TILE_PT_TO_PT_CUH__

#include <cuda_runtime.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "device_host/nvshmem_tensor.h"
#include "non_abi/device/common/nvshmemi_tile_utils.cuh"

#ifdef __CUDA_ARCH__
// Tile APIs need C++ 17
#if defined(__cplusplus) && __cplusplus >= 201703L

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_cpy_threadgroup_v4(
    int4 *dest, const int4 *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /* src_stride_major_dim == 1 && dst_stride_major_dim == 1 for vectorized implementation */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems */
    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                    dest + ((j) % nelem_major_dim) +
                    (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                : "memory");
        }
    } else {
        using vtype = int4;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            /* nelem_major_dim is in vector units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u & elem_coord_major in elemType units
                     */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /* end of if else */
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_cpy_threadgroup_v2(
    uint64_t *dest, const uint64_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /* src_stride_major_dim == 1 && dst_stride_major_dim == 1 for vectorized implementation */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/

    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            asm("ld.global.b64 %0, [%1];"
                : "=l"(val1)
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("st.global.u64 [%0], %1;" ::"l"(dest + ((j) % nelem_major_dim) +
                                                (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "l"(val1)
                : "memory");
        }
    } else {
        using vtype = uint64_t;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            /* nelem_major_dim is in vector units */
            /* compute elem_coord_major in elemType units */
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.b64 %0, [%1];"
                    : "=l"(val1)
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("st.global.u64 [%0], %1;" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "l"(val1)
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u and elem_coord_major in elemType
                     * units */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /* end of if else */
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_cpy_threadgroup_v1(
    uint32_t *dest, const uint32_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems */
    using vtype = uint32_t;
    using cxx_type = uint32_t;
    if constexpr (std::is_empty<tuple_t>::value) {
        cxx_type val;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            asm("ld.global.b32 %0, [%1];"
                : "=r"(val)
                : "l"(source + ((j % nelem_major_dim) * src_stride_major_dim) +
                      ((j / nelem_major_dim) * src_stride_minor_dim)));

            asm("st.global.u32 [%0], %1;" ::"l"(dest +
                                                ((j % nelem_major_dim) * dst_stride_major_dim) +
                                                ((j / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(val)
                : "memory");
        }
    } else {
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            /* nelem_major_dim is in vector units */
            /* compute elem_coord_major in elemType units */
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);
            cxx_type val;

            /* convert elem_coord_major from elemType to cxx_type units */
            /* no change to elem_coord_minor */
            elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

            for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                /* check if elem is within boundary, use u and elem_coord_major in elemType units */
                if (is_less_than<tuple_t, major_dim>(
                        start_coord,
                        create_coord_tuple<major_dim>(
                            ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                            elem_coord_minor),
                        boundary)) {
                    /* convert strides from vector to cxx_type units */
                    asm("ld.global.b32 %0, [%1];"
                        : "=r"(val)
                        : "l"(reinterpret_cast<const cxx_type *>(source) + (elem_coord_major + u) +
                              (elem_coord_minor * src_stride_minor_dim *
                               (sizeof(vtype) / sizeof(cxx_type)))));

                    asm("st.global.u32 [%0], %1;" ::"l"(reinterpret_cast<cxx_type *>(dest) +
                                                        (elem_coord_major + u) +
                                                        (elem_coord_minor * dst_stride_minor_dim *
                                                         (sizeof(vtype) / sizeof(cxx_type)))),
                        "r"(val)
                        : "memory");
                }
            }
        }
    } /* end of if else */
}

// Select implementation based on the operation, datatype
template <typename vtype, typename T, threadgroup_t scope, typename tuple_t, bool is_put,
          int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_cpy_nvl_threadgroup_vec(
    T *src, T *dst, const int size_major_dim, const int size_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary, int pe) {
    vtype *src_v, *dst_v;
    if constexpr (is_put) {
        // src is local and dst is unicast P2P address
        src_v = reinterpret_cast<vtype *>(src);
        dst_v = reinterpret_cast<vtype *>(nvshmem_ptr(dst, pe));
        assert((dst_v != nullptr) &&
               "Failed to get nvlink ptr for "
               "destination, destination address is not a valid nvlink address");
    } else {
        // dst is local and src is unicast P2P address
        dst_v = reinterpret_cast<vtype *>(dst);
        src_v = reinterpret_cast<vtype *>(nvshmem_ptr(src, pe));
        assert((src_v != nullptr) &&
               "Failed to get nvlink ptr for "
               "source, source address is not a valid nvlink address");
    }

    int src_stride_minor_dim_v = src_stride_minor_dim;
    if (src_stride_minor_dim > 1) {
        src_stride_minor_dim_v = (src_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_minor_dim_v = dst_stride_minor_dim;
    if (dst_stride_minor_dim > 1) {
        dst_stride_minor_dim_v = (dst_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int src_stride_major_dim_v = src_stride_major_dim;  // keep stride as is if ==1
    if (src_stride_major_dim > 1) {
        src_stride_major_dim_v = (src_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_major_dim_v = dst_stride_major_dim;
    if (dst_stride_major_dim > 1) {
        dst_stride_major_dim_v = (dst_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }

    int nelem_major_dim = (size_major_dim * sizeof(T)) / sizeof(vtype);
    int nelem_minor_dim = size_minor_dim;

    if constexpr (std::is_same<vtype, int4>::value) {
        nvshmemi_tile_cpy_threadgroup_v4<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint64_t>::value) {
        nvshmemi_tile_cpy_threadgroup_v2<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint32_t>::value) {
        nvshmemi_tile_cpy_threadgroup_v1<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);
    }
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          bool is_put, int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_cpy_nvl_threadgroup_dim(src_tensor_t src_tensor,
                                                             dst_tensor_t dst_tensor,
                                                             tuple_t start_coord, tuple_t boundary,
                                                             int pe) {
    using T = typename src_tensor_t::value_type;

    // check for vector len == 4
    // Conditions: ptr must be aligned to int4, shape must be a multiple of 16, stride must be a
    // multiple of 16
    if (((size_t)src_tensor.data() % sizeof(int4) == 0) &&
        ((size_t)dst_tensor.data() % sizeof(int4) == 0) &&
        (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(int4)) == 0)) {
        nvshmemi_tile_cpy_nvl_threadgroup_vec<int4, T, scope, tuple_t, is_put, major_dim,
                                              minor_dim>(
            src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary, pe);

    } else if (((size_t)src_tensor.data() % sizeof(uint64_t) == 0) &&
               ((size_t)dst_tensor.data() % sizeof(uint64_t) == 0) &&
               (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0)) {
        nvshmemi_tile_cpy_nvl_threadgroup_vec<uint64_t, T, scope, tuple_t, is_put, major_dim,
                                              minor_dim>(
            src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary, pe);

    } else {  // vector len 1
        nvshmemi_tile_cpy_nvl_threadgroup_vec<uint32_t, T, scope, tuple_t, is_put, major_dim,
                                              minor_dim>(
            src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary, pe);
    }
}

// specialize for the vectorization
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          bool is_put>
__device__ inline void nvshmemi_tile_cpy_nvl_threadgroup(src_tensor_t src_tensor,
                                                         dst_tensor_t dst_tensor,
                                                         tuple_t start_coord, tuple_t boundary,
                                                         int pe) {
    using T = typename src_tensor_t::value_type;
    if constexpr ((get_constant(safe_get<0>(decltype(src_tensor.stride()){})) == 1) &&
                  (get_constant(safe_get<0>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 0 major
        constexpr int major_dim = 0;
        constexpr int minor_dim = 1;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_cpy_nvl_threadgroup_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, is_put,
                                              major_dim, minor_dim>(src_tensor, dst_tensor,
                                                                    start_coord, boundary, pe);
    } else if constexpr ((get_constant(safe_get<1>(decltype(src_tensor.stride()){})) == 1) &&
                         (get_constant(safe_get<1>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 1 major
        constexpr int major_dim = 1;
        constexpr int minor_dim = 0;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_cpy_nvl_threadgroup_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, is_put,
                                              major_dim, minor_dim>(src_tensor, dst_tensor,
                                                                    start_coord, boundary, pe);
    } else {
        // No contiguous dimension found at compile time
        // TODO support when major dimension for src and dst are different
        if ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1)) {
            constexpr int major_dim = 1;
            constexpr int minor_dim = 0;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }

            nvshmemi_tile_cpy_nvl_threadgroup_dim<src_tensor_t, dst_tensor_t, tuple_t, scope,
                                                  is_put, major_dim, minor_dim>(
                src_tensor, dst_tensor, start_coord, boundary, pe);
        } else {
            // setting major_dim to 0, minor_dim to 1
            constexpr int major_dim = 0;
            constexpr int minor_dim = 1;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }
            nvshmemi_tile_cpy_nvl_threadgroup_dim<src_tensor_t, dst_tensor_t, tuple_t, scope,
                                                  is_put, major_dim, minor_dim>(
                src_tensor, dst_tensor, start_coord, boundary, pe);
        }
    }
}

// Remote copy
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          bool is_put, int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_cpy_put_get_threadgroup_wrapper(src_tensor_t src_tensor,
                                                                     dst_tensor_t dst_tensor,
                                                                     tuple_t start_coord,
                                                                     tuple_t boundary, int pe) {
    using T = typename src_tensor_t::value_type;

    // issue one (thread scope) PUT/GET along major dimension (stride 1)
    T *src_ptr = src_tensor.data();
    T *dst_ptr = dst_tensor.data();
    const int myIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
    const int groupSize = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();

    for (int i = myIdx; i < get_shape_element<minor_dim>(src_tensor); i += groupSize) {
        dst_ptr = dst_tensor.data() + get_stride_element<minor_dim>(dst_tensor) * i;
        src_ptr = src_tensor.data() + get_stride_element<minor_dim>(src_tensor) * i;
        if constexpr (is_put) {
            nvshmemi_put_nbi<T, NVSHMEMI_THREADGROUP_THREAD>(
                dst_ptr, src_ptr, get_shape_element<major_dim>(src_tensor), pe);
        } else {
            nvshmemi_get_nbi<T, NVSHMEMI_THREADGROUP_THREAD>(
                dst_ptr, src_ptr, get_shape_element<major_dim>(src_tensor), pe);
        }
    }
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          bool is_put>
__device__ inline void nvshmemi_tile_cpy_remote_threadgroup(src_tensor_t src_tensor,
                                                            dst_tensor_t dst_tensor,
                                                            tuple_t start_coord, tuple_t boundary,
                                                            int pe) {
    if constexpr ((get_constant(safe_get<0>(decltype(src_tensor.stride()){})) == 1) &&
                  (get_constant(safe_get<0>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 0 major
        constexpr int major_dim = 0;
        constexpr int minor_dim = 1;

        nvshmemi_tile_cpy_put_get_threadgroup_wrapper<src_tensor_t, dst_tensor_t, tuple_t, scope,
                                                      is_put, major_dim, minor_dim>(
            src_tensor, dst_tensor, start_coord, boundary, pe);
    } else if constexpr ((get_constant(safe_get<1>(decltype(src_tensor.stride()){})) == 1) &&
                         (get_constant(safe_get<1>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 1 major
        constexpr int major_dim = 1;
        constexpr int minor_dim = 0;

        nvshmemi_tile_cpy_put_get_threadgroup_wrapper<src_tensor_t, dst_tensor_t, tuple_t, scope,
                                                      is_put, major_dim, minor_dim>(
            src_tensor, dst_tensor, start_coord, boundary, pe);
    } else {
        // No contiguous dimension found at compile time
        // TODO support when major dimension for src and dst are different
        if ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1)) {
            constexpr int major_dim = 1;
            constexpr int minor_dim = 0;

            nvshmemi_tile_cpy_put_get_threadgroup_wrapper<src_tensor_t, dst_tensor_t, tuple_t,
                                                          scope, is_put, major_dim, minor_dim>(
                src_tensor, dst_tensor, start_coord, boundary, pe);
        } else {
            // setting major_dim to 0, minor_dim to 1
            constexpr int major_dim = 0;
            constexpr int minor_dim = 1;

            nvshmemi_tile_cpy_put_get_threadgroup_wrapper<src_tensor_t, dst_tensor_t, tuple_t,
                                                          scope, is_put, major_dim, minor_dim>(
                src_tensor, dst_tensor, start_coord, boundary, pe);
        }
    }
}

#endif  // __cplusplus >= 201703L
// Tile Put entrypoint
// Call underlying function based on scope and algo
template <nvshmemx::tile_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope>
__device__ inline int nvshmemi_tile_put(src_tensor_t src_tensor, dst_tensor_t dst_tensor,
                                        tuple_t start_coord, tuple_t boundary, int pe,
                                        uint64_t flag) {
#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
    return -1;
#else
    using T = typename src_tensor_t::value_type;

    static_assert(
        std::is_same<typename src_tensor_t::value_type, typename dst_tensor_t::value_type>::value,
        "Source and destination tensors must have the same type");

    static_assert((algo == nvshmemx::tile_algo_t::PEER_PUSH_NBI) ||
                      (algo == nvshmemx::tile_algo_t::REMOTE_PUSH_NBI),
                  "Unsupported tile Put algorithm. "
                  "Currently PEER_PUSH_NBI and REMOTE_PUSH_NBI are supported for tile put");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check shape
    assert(get_tensor_size(src_tensor) == get_tensor_size(dst_tensor));

    // TODO add other data types
    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value) ||
                   (is_cutlass_half<T>()) || (is_cutlass_bfloat<T>)),
                  "Unsupported datatype");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are atleast continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    // if NvLink based
    if constexpr (algo == nvshmemx::tile_algo_t::PEER_PUSH_NBI) {
        // User should ensure src data is ready
        nvshmemi_tile_cpy_nvl_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope, 1 /*is_put*/>(
            src_tensor, dst_tensor, start_coord, boundary, pe);

    } else if constexpr (algo == nvshmemx::tile_algo_t::REMOTE_PUSH_NBI) {
        nvshmemi_tile_cpy_remote_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope,
                                             1 /*is_put*/>(src_tensor, dst_tensor, start_coord,
                                                           boundary, pe);

    } else {
        // Extend as other algorithms are added
    }
    return 0;
#endif  // __cplusplus >= 201703L
}

// Tile Get entrypoint
// Call underlying function based on scope and algo
template <nvshmemx::tile_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope>
__device__ inline int nvshmemi_tile_get(src_tensor_t src_tensor, dst_tensor_t dst_tensor,
                                        tuple_t start_coord, tuple_t boundary, int pe,
                                        uint64_t flag) {
#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
    return -1;
#else
    using T = typename src_tensor_t::value_type;

    static_assert(
        std::is_same<typename src_tensor_t::value_type, typename dst_tensor_t::value_type>::value,
        "Source and destination tensors must have the same type");

    static_assert((algo == nvshmemx::tile_algo_t::PEER_PULL_NBI) ||
                      (algo == nvshmemx::tile_algo_t::REMOTE_PULL_NBI),
                  "Unsupported tile Get algorithm. "
                  "Currently PEER_PULL_NBI and REMOTE_PULL_NBI are supported for tile get");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check shape
    assert(get_tensor_size(src_tensor) == get_tensor_size(dst_tensor));

    // TODO add other data types
    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value) ||
                   (is_cutlass_half<T>()) || (is_cutlass_bfloat<T>)),
                  "Unsupported datatype");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are atleast continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    // if NvLink based
    if constexpr (algo == nvshmemx::tile_algo_t::PEER_PULL_NBI) {
        // User should ensure src data is ready
        nvshmemi_tile_cpy_nvl_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope, 0>(
            src_tensor, dst_tensor, start_coord, boundary, pe);

    } else if constexpr (algo == nvshmemx::tile_algo_t::REMOTE_PULL_NBI) {
        nvshmemi_tile_cpy_remote_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope, 0>(
            src_tensor, dst_tensor, start_coord, boundary, pe);

    } else {
        // Extend as other algorithms are added
    }
    return 0;
#endif  // __cplusplus >= 201703L
}

#endif /* __CUDA_ARCH__ */
#endif /* __TILE_PT_TO_PT_CUH__ */
