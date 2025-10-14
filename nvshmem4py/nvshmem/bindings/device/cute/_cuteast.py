
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information
from __future__ import annotations

import cutlass
from cutlass import cute
from cutlass.cute.typing import AddressSpace

try:
    import cutlass._mlir.dialects.cute as _cute_ir
except Exception:  # pragma: no cover - depends on CuTe runtime
    _cute_ir = None

class _CutePtrType:
    def __init__(self, dtype, address_space=AddressSpace.gmem, alignment=None):
        self._dtype = dtype
        self._address_space = address_space
        self._alignment = alignment

    def __get_mlir_types__(self):
        if _cute_ir is None:
            raise RuntimeError("CuTe MLIR context is not available for pointer type generation")
        if self._alignment is None:
            align = getattr(self._dtype, 'width', 8) // 8 or 1
        else:
            align = self._alignment
        return [_cute_ir.PtrType.get(self._dtype.mlir_type, self._address_space, align)]

team_my_pe = cute.ffi(name="nvshmem_team_my_pe", params_types=[cutlass.Int32], return_type=cutlass.Int32)

team_n_pes = cute.ffi(name="nvshmem_team_n_pes", params_types=[cutlass.Int32], return_type=cutlass.Int32)

team_translate_pe = cute.ffi(name="nvshmem_team_translate_pe", params_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

my_pe = cute.ffi(name="nvshmem_my_pe", params_types=[], return_type=cutlass.Int32)

n_pes = cute.ffi(name="nvshmem_n_pes", params_types=[], return_type=cutlass.Int32)

info_get_name = cute.ffi(name="nvshmem_info_get_name", params_types=[_CutePtrType(cutlass.Int8)])

info_get_version = cute.ffi(name="nvshmem_info_get_version", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32)])

bfloat16_p = cute.ffi(name="nvshmem_bfloat16_p", params_types=[_CutePtrType(cutlass.BFloat16), cutlass.BFloat16, cutlass.Int32])

half_p = cute.ffi(name="nvshmem_half_p", params_types=[_CutePtrType(cutlass.Float16), cutlass.Float16, cutlass.Int32])

float_p = cute.ffi(name="nvshmem_float_p", params_types=[_CutePtrType(cutlass.Float32), cutlass.Float32, cutlass.Int32])

double_p = cute.ffi(name="nvshmem_double_p", params_types=[_CutePtrType(cutlass.Float64), cutlass.Float64, cutlass.Int32])

char_p = cute.ffi(name="nvshmem_char_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32])

short_p = cute.ffi(name="nvshmem_short_p", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int16, cutlass.Int32])

schar_p = cute.ffi(name="nvshmem_schar_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32])

int_p = cute.ffi(name="nvshmem_int_p", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

long_p = cute.ffi(name="nvshmem_long_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

longlong_p = cute.ffi(name="nvshmem_longlong_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uchar_p = cute.ffi(name="nvshmem_uchar_p", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Uint8, cutlass.Int32])

ushort_p = cute.ffi(name="nvshmem_ushort_p", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint16, cutlass.Int32])

uint_p = cute.ffi(name="nvshmem_uint_p", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_p = cute.ffi(name="nvshmem_ulong_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_p = cute.ffi(name="nvshmem_ulonglong_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_p = cute.ffi(name="nvshmem_int8_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32])

int16_p = cute.ffi(name="nvshmem_int16_p", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int16, cutlass.Int32])

int32_p = cute.ffi(name="nvshmem_int32_p", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_p = cute.ffi(name="nvshmem_int64_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint8_p = cute.ffi(name="nvshmem_uint8_p", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Uint8, cutlass.Int32])

uint16_p = cute.ffi(name="nvshmem_uint16_p", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint16, cutlass.Int32])

uint32_p = cute.ffi(name="nvshmem_uint32_p", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_p = cute.ffi(name="nvshmem_uint64_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_p = cute.ffi(name="nvshmem_size_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_p = cute.ffi(name="nvshmem_ptrdiff_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

bfloat16_g = cute.ffi(name="nvshmem_bfloat16_g", params_types=[_CutePtrType(cutlass.BFloat16), cutlass.Int32], return_type=cutlass.BFloat16)

half_g = cute.ffi(name="nvshmem_half_g", params_types=[_CutePtrType(cutlass.Float16), cutlass.Int32], return_type=cutlass.Float16)

float_g = cute.ffi(name="nvshmem_float_g", params_types=[_CutePtrType(cutlass.Float32), cutlass.Int32], return_type=cutlass.Float32)

double_g = cute.ffi(name="nvshmem_double_g", params_types=[_CutePtrType(cutlass.Float64), cutlass.Int32], return_type=cutlass.Float64)

char_g = cute.ffi(name="nvshmem_char_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32], return_type=cutlass.Int8)

short_g = cute.ffi(name="nvshmem_short_g", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32], return_type=cutlass.Int16)

schar_g = cute.ffi(name="nvshmem_schar_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32], return_type=cutlass.Int8)

int_g = cute.ffi(name="nvshmem_int_g", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

long_g = cute.ffi(name="nvshmem_long_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

longlong_g = cute.ffi(name="nvshmem_longlong_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

uchar_g = cute.ffi(name="nvshmem_uchar_g", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Int32], return_type=cutlass.Uint8)

ushort_g = cute.ffi(name="nvshmem_ushort_g", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32], return_type=cutlass.Uint16)

uint_g = cute.ffi(name="nvshmem_uint_g", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

ulong_g = cute.ffi(name="nvshmem_ulong_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_g = cute.ffi(name="nvshmem_ulonglong_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

int8_g = cute.ffi(name="nvshmem_int8_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32], return_type=cutlass.Int8)

int16_g = cute.ffi(name="nvshmem_int16_g", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32], return_type=cutlass.Int16)

int32_g = cute.ffi(name="nvshmem_int32_g", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

int64_g = cute.ffi(name="nvshmem_int64_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

uint8_g = cute.ffi(name="nvshmem_uint8_g", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Int32], return_type=cutlass.Uint8)

uint16_g = cute.ffi(name="nvshmem_uint16_g", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32], return_type=cutlass.Uint16)

uint32_g = cute.ffi(name="nvshmem_uint32_g", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

uint64_g = cute.ffi(name="nvshmem_uint64_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

size_g = cute.ffi(name="nvshmem_size_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

ptrdiff_g = cute.ffi(name="nvshmem_ptrdiff_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

bfloat16_put = cute.ffi(name="nvshmem_bfloat16_put", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_put = cute.ffi(name="nvshmem_half_put", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_put = cute.ffi(name="nvshmem_float_put", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_put = cute.ffi(name="nvshmem_double_put", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_put = cute.ffi(name="nvshmem_char_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_put = cute.ffi(name="nvshmem_short_put", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_put = cute.ffi(name="nvshmem_schar_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_put = cute.ffi(name="nvshmem_int_put", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_put = cute.ffi(name="nvshmem_long_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put = cute.ffi(name="nvshmem_longlong_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_put = cute.ffi(name="nvshmem_uchar_put", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_put = cute.ffi(name="nvshmem_ushort_put", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_put = cute.ffi(name="nvshmem_uint_put", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_put = cute.ffi(name="nvshmem_ulong_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put = cute.ffi(name="nvshmem_ulonglong_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_put = cute.ffi(name="nvshmem_int8_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_put = cute.ffi(name="nvshmem_int16_put", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_put = cute.ffi(name="nvshmem_int32_put", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_put = cute.ffi(name="nvshmem_int64_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_put = cute.ffi(name="nvshmem_uint8_put", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_put = cute.ffi(name="nvshmem_uint16_put", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_put = cute.ffi(name="nvshmem_uint32_put", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_put = cute.ffi(name="nvshmem_uint64_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put = cute.ffi(name="nvshmem_size_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put = cute.ffi(name="nvshmem_ptrdiff_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

bfloat16_put_signal = cute.ffi(name="nvshmem_bfloat16_put_signal", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal = cute.ffi(name="nvshmem_half_put_signal", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal = cute.ffi(name="nvshmem_float_put_signal", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal = cute.ffi(name="nvshmem_double_put_signal", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal = cute.ffi(name="nvshmem_char_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal = cute.ffi(name="nvshmem_short_put_signal", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal = cute.ffi(name="nvshmem_schar_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal = cute.ffi(name="nvshmem_int_put_signal", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal = cute.ffi(name="nvshmem_long_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal = cute.ffi(name="nvshmem_longlong_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal = cute.ffi(name="nvshmem_uchar_put_signal", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal = cute.ffi(name="nvshmem_ushort_put_signal", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal = cute.ffi(name="nvshmem_uint_put_signal", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal = cute.ffi(name="nvshmem_ulong_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal = cute.ffi(name="nvshmem_ulonglong_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal = cute.ffi(name="nvshmem_int8_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal = cute.ffi(name="nvshmem_int16_put_signal", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal = cute.ffi(name="nvshmem_int32_put_signal", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal = cute.ffi(name="nvshmem_int64_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal = cute.ffi(name="nvshmem_uint8_put_signal", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal = cute.ffi(name="nvshmem_uint16_put_signal", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal = cute.ffi(name="nvshmem_uint32_put_signal", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal = cute.ffi(name="nvshmem_uint64_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal = cute.ffi(name="nvshmem_size_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal = cute.ffi(name="nvshmem_ptrdiff_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_get = cute.ffi(name="nvshmem_bfloat16_get", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_get = cute.ffi(name="nvshmem_half_get", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_get = cute.ffi(name="nvshmem_float_get", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_get = cute.ffi(name="nvshmem_double_get", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_get = cute.ffi(name="nvshmem_char_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_get = cute.ffi(name="nvshmem_short_get", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_get = cute.ffi(name="nvshmem_schar_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_get = cute.ffi(name="nvshmem_int_get", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_get = cute.ffi(name="nvshmem_long_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get = cute.ffi(name="nvshmem_longlong_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_get = cute.ffi(name="nvshmem_uchar_get", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_get = cute.ffi(name="nvshmem_ushort_get", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_get = cute.ffi(name="nvshmem_uint_get", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_get = cute.ffi(name="nvshmem_ulong_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get = cute.ffi(name="nvshmem_ulonglong_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_get = cute.ffi(name="nvshmem_int8_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_get = cute.ffi(name="nvshmem_int16_get", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_get = cute.ffi(name="nvshmem_int32_get", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_get = cute.ffi(name="nvshmem_int64_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_get = cute.ffi(name="nvshmem_uint8_get", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_get = cute.ffi(name="nvshmem_uint16_get", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_get = cute.ffi(name="nvshmem_uint32_get", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_get = cute.ffi(name="nvshmem_uint64_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get = cute.ffi(name="nvshmem_size_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get = cute.ffi(name="nvshmem_ptrdiff_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

put8 = cute.ffi(name="nvshmem_put8", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16 = cute.ffi(name="nvshmem_put16", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

put32 = cute.ffi(name="nvshmem_put32", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

put64 = cute.ffi(name="nvshmem_put64", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

put128 = cute.ffi(name="nvshmem_put128", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_signal = cute.ffi(name="nvshmem_putmem_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal = cute.ffi(name="nvshmem_put8_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put16_signal = cute.ffi(name="nvshmem_put16_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal = cute.ffi(name="nvshmem_put32_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal = cute.ffi(name="nvshmem_put64_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal = cute.ffi(name="nvshmem_put128_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

get8 = cute.ffi(name="nvshmem_get8", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16 = cute.ffi(name="nvshmem_get16", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

get32 = cute.ffi(name="nvshmem_get32", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

get64 = cute.ffi(name="nvshmem_get64", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

get128 = cute.ffi(name="nvshmem_get128", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem = cute.ffi(name="nvshmem_putmem", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

getmem = cute.ffi(name="nvshmem_getmem", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

bfloat16_put_nbi = cute.ffi(name="nvshmem_bfloat16_put_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_put_nbi = cute.ffi(name="nvshmem_half_put_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_put_nbi = cute.ffi(name="nvshmem_float_put_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_put_nbi = cute.ffi(name="nvshmem_double_put_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_put_nbi = cute.ffi(name="nvshmem_char_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_put_nbi = cute.ffi(name="nvshmem_short_put_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_put_nbi = cute.ffi(name="nvshmem_schar_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_put_nbi = cute.ffi(name="nvshmem_int_put_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_put_nbi = cute.ffi(name="nvshmem_long_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put_nbi = cute.ffi(name="nvshmem_longlong_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_put_nbi = cute.ffi(name="nvshmem_uchar_put_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_put_nbi = cute.ffi(name="nvshmem_ushort_put_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_put_nbi = cute.ffi(name="nvshmem_uint_put_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_put_nbi = cute.ffi(name="nvshmem_ulong_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put_nbi = cute.ffi(name="nvshmem_ulonglong_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_put_nbi = cute.ffi(name="nvshmem_int8_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_put_nbi = cute.ffi(name="nvshmem_int16_put_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_put_nbi = cute.ffi(name="nvshmem_int32_put_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_put_nbi = cute.ffi(name="nvshmem_int64_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_put_nbi = cute.ffi(name="nvshmem_uint8_put_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_put_nbi = cute.ffi(name="nvshmem_uint16_put_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_put_nbi = cute.ffi(name="nvshmem_uint32_put_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_put_nbi = cute.ffi(name="nvshmem_uint64_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put_nbi = cute.ffi(name="nvshmem_size_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put_nbi = cute.ffi(name="nvshmem_ptrdiff_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

bfloat16_put_signal_nbi = cute.ffi(name="nvshmem_bfloat16_put_signal_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal_nbi = cute.ffi(name="nvshmem_half_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal_nbi = cute.ffi(name="nvshmem_float_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal_nbi = cute.ffi(name="nvshmem_double_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal_nbi = cute.ffi(name="nvshmem_char_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal_nbi = cute.ffi(name="nvshmem_short_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal_nbi = cute.ffi(name="nvshmem_schar_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal_nbi = cute.ffi(name="nvshmem_int_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal_nbi = cute.ffi(name="nvshmem_long_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal_nbi = cute.ffi(name="nvshmem_longlong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal_nbi = cute.ffi(name="nvshmem_uchar_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal_nbi = cute.ffi(name="nvshmem_ushort_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal_nbi = cute.ffi(name="nvshmem_uint_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal_nbi = cute.ffi(name="nvshmem_ulong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal_nbi = cute.ffi(name="nvshmem_ulonglong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal_nbi = cute.ffi(name="nvshmem_int8_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal_nbi = cute.ffi(name="nvshmem_int16_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal_nbi = cute.ffi(name="nvshmem_int32_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal_nbi = cute.ffi(name="nvshmem_int64_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal_nbi = cute.ffi(name="nvshmem_uint8_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal_nbi = cute.ffi(name="nvshmem_uint16_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal_nbi = cute.ffi(name="nvshmem_uint32_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal_nbi = cute.ffi(name="nvshmem_uint64_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal_nbi = cute.ffi(name="nvshmem_size_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal_nbi = cute.ffi(name="nvshmem_ptrdiff_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_get_nbi = cute.ffi(name="nvshmem_bfloat16_get_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_get_nbi = cute.ffi(name="nvshmem_half_get_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_get_nbi = cute.ffi(name="nvshmem_float_get_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_get_nbi = cute.ffi(name="nvshmem_double_get_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_get_nbi = cute.ffi(name="nvshmem_char_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_get_nbi = cute.ffi(name="nvshmem_short_get_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_get_nbi = cute.ffi(name="nvshmem_schar_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_get_nbi = cute.ffi(name="nvshmem_int_get_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_get_nbi = cute.ffi(name="nvshmem_long_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get_nbi = cute.ffi(name="nvshmem_longlong_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_get_nbi = cute.ffi(name="nvshmem_uchar_get_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_get_nbi = cute.ffi(name="nvshmem_ushort_get_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_get_nbi = cute.ffi(name="nvshmem_uint_get_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_get_nbi = cute.ffi(name="nvshmem_ulong_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get_nbi = cute.ffi(name="nvshmem_ulonglong_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_get_nbi = cute.ffi(name="nvshmem_int8_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_get_nbi = cute.ffi(name="nvshmem_int16_get_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_get_nbi = cute.ffi(name="nvshmem_int32_get_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_get_nbi = cute.ffi(name="nvshmem_int64_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_get_nbi = cute.ffi(name="nvshmem_uint8_get_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_get_nbi = cute.ffi(name="nvshmem_uint16_get_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_get_nbi = cute.ffi(name="nvshmem_uint32_get_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_get_nbi = cute.ffi(name="nvshmem_uint64_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get_nbi = cute.ffi(name="nvshmem_size_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get_nbi = cute.ffi(name="nvshmem_ptrdiff_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

put8_nbi = cute.ffi(name="nvshmem_put8_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16_nbi = cute.ffi(name="nvshmem_put16_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put32_nbi = cute.ffi(name="nvshmem_put32_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put64_nbi = cute.ffi(name="nvshmem_put64_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put128_nbi = cute.ffi(name="nvshmem_put128_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get8_nbi = cute.ffi(name="nvshmem_get8_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16_nbi = cute.ffi(name="nvshmem_get16_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get32_nbi = cute.ffi(name="nvshmem_get32_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get64_nbi = cute.ffi(name="nvshmem_get64_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get128_nbi = cute.ffi(name="nvshmem_get128_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_nbi = cute.ffi(name="nvshmem_putmem_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_signal_nbi = cute.ffi(name="nvshmem_putmem_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal_nbi = cute.ffi(name="nvshmem_put8_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put16_signal_nbi = cute.ffi(name="nvshmem_put16_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal_nbi = cute.ffi(name="nvshmem_put32_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal_nbi = cute.ffi(name="nvshmem_put64_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal_nbi = cute.ffi(name="nvshmem_put128_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

getmem_nbi = cute.ffi(name="nvshmem_getmem_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

bfloat16_iput = cute.ffi(name="nvshmem_bfloat16_iput", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iput = cute.ffi(name="nvshmem_half_iput", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iput = cute.ffi(name="nvshmem_float_iput", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iput = cute.ffi(name="nvshmem_double_iput", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iput = cute.ffi(name="nvshmem_char_iput", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iput = cute.ffi(name="nvshmem_short_iput", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iput = cute.ffi(name="nvshmem_schar_iput", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iput = cute.ffi(name="nvshmem_int_iput", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iput = cute.ffi(name="nvshmem_long_iput", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iput = cute.ffi(name="nvshmem_longlong_iput", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iput = cute.ffi(name="nvshmem_uchar_iput", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iput = cute.ffi(name="nvshmem_ushort_iput", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iput = cute.ffi(name="nvshmem_uint_iput", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iput = cute.ffi(name="nvshmem_ulong_iput", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iput = cute.ffi(name="nvshmem_ulonglong_iput", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iput = cute.ffi(name="nvshmem_int8_iput", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iput = cute.ffi(name="nvshmem_int16_iput", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iput = cute.ffi(name="nvshmem_int32_iput", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iput = cute.ffi(name="nvshmem_int64_iput", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iput = cute.ffi(name="nvshmem_uint8_iput", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iput = cute.ffi(name="nvshmem_uint16_iput", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iput = cute.ffi(name="nvshmem_uint32_iput", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iput = cute.ffi(name="nvshmem_uint64_iput", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iput = cute.ffi(name="nvshmem_size_iput", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iput = cute.ffi(name="nvshmem_ptrdiff_iput", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput8 = cute.ffi(name="nvshmem_iput8", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput16 = cute.ffi(name="nvshmem_iput16", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput32 = cute.ffi(name="nvshmem_iput32", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput64 = cute.ffi(name="nvshmem_iput64", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput128 = cute.ffi(name="nvshmem_iput128", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

bfloat16_iget = cute.ffi(name="nvshmem_bfloat16_iget", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iget = cute.ffi(name="nvshmem_half_iget", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iget = cute.ffi(name="nvshmem_float_iget", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iget = cute.ffi(name="nvshmem_double_iget", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iget = cute.ffi(name="nvshmem_char_iget", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iget = cute.ffi(name="nvshmem_short_iget", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iget = cute.ffi(name="nvshmem_schar_iget", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iget = cute.ffi(name="nvshmem_int_iget", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iget = cute.ffi(name="nvshmem_long_iget", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iget = cute.ffi(name="nvshmem_longlong_iget", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iget = cute.ffi(name="nvshmem_uchar_iget", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iget = cute.ffi(name="nvshmem_ushort_iget", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iget = cute.ffi(name="nvshmem_uint_iget", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iget = cute.ffi(name="nvshmem_ulong_iget", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iget = cute.ffi(name="nvshmem_ulonglong_iget", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iget = cute.ffi(name="nvshmem_int8_iget", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iget = cute.ffi(name="nvshmem_int16_iget", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iget = cute.ffi(name="nvshmem_int32_iget", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iget = cute.ffi(name="nvshmem_int64_iget", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iget = cute.ffi(name="nvshmem_uint8_iget", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iget = cute.ffi(name="nvshmem_uint16_iget", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iget = cute.ffi(name="nvshmem_uint32_iget", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iget = cute.ffi(name="nvshmem_uint64_iget", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iget = cute.ffi(name="nvshmem_size_iget", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iget = cute.ffi(name="nvshmem_ptrdiff_iget", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget8 = cute.ffi(name="nvshmem_iget8", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget16 = cute.ffi(name="nvshmem_iget16", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget32 = cute.ffi(name="nvshmem_iget32", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget64 = cute.ffi(name="nvshmem_iget64", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget128 = cute.ffi(name="nvshmem_iget128", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_test = cute.ffi(name="nvshmem_short_test", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32, cutlass.Int16], return_type=cutlass.Int32)

int_test = cute.ffi(name="nvshmem_int_test", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

long_test = cute.ffi(name="nvshmem_long_test", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

longlong_test = cute.ffi(name="nvshmem_longlong_test", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

ushort_test = cute.ffi(name="nvshmem_ushort_test", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Int32)

uint_test = cute.ffi(name="nvshmem_uint_test", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Int32)

ulong_test = cute.ffi(name="nvshmem_ulong_test", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_test = cute.ffi(name="nvshmem_ulonglong_test", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

int32_test = cute.ffi(name="nvshmem_int32_test", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_test = cute.ffi(name="nvshmem_int64_test", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

uint32_test = cute.ffi(name="nvshmem_uint32_test", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Int32)

uint64_test = cute.ffi(name="nvshmem_uint64_test", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

size_test = cute.ffi(name="nvshmem_size_test", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_test = cute.ffi(name="nvshmem_ptrdiff_test", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

short_test_all = cute.ffi(name="nvshmem_short_test_all", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16], return_type=cutlass.Int32)

int_test_all = cute.ffi(name="nvshmem_int_test_all", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

long_test_all = cute.ffi(name="nvshmem_long_test_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

longlong_test_all = cute.ffi(name="nvshmem_longlong_test_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

ushort_test_all = cute.ffi(name="nvshmem_ushort_test_all", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Int32)

uint_test_all = cute.ffi(name="nvshmem_uint_test_all", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Int32)

ulong_test_all = cute.ffi(name="nvshmem_ulong_test_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_test_all = cute.ffi(name="nvshmem_ulonglong_test_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

int32_test_all = cute.ffi(name="nvshmem_int32_test_all", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_test_all = cute.ffi(name="nvshmem_int64_test_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

uint32_test_all = cute.ffi(name="nvshmem_uint32_test_all", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Int32)

uint64_test_all = cute.ffi(name="nvshmem_uint64_test_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

size_test_all = cute.ffi(name="nvshmem_size_test_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_test_all = cute.ffi(name="nvshmem_ptrdiff_test_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Int32)

short_test_any = cute.ffi(name="nvshmem_short_test_any", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16], return_type=cutlass.Uint64)

int_test_any = cute.ffi(name="nvshmem_int_test_any", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

long_test_any = cute.ffi(name="nvshmem_long_test_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

longlong_test_any = cute.ffi(name="nvshmem_longlong_test_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

ushort_test_any = cute.ffi(name="nvshmem_ushort_test_any", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Uint64)

uint_test_any = cute.ffi(name="nvshmem_uint_test_any", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

ulong_test_any = cute.ffi(name="nvshmem_ulong_test_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ulonglong_test_any = cute.ffi(name="nvshmem_ulonglong_test_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

int32_test_any = cute.ffi(name="nvshmem_int32_test_any", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

int64_test_any = cute.ffi(name="nvshmem_int64_test_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

uint32_test_any = cute.ffi(name="nvshmem_uint32_test_any", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

uint64_test_any = cute.ffi(name="nvshmem_uint64_test_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

size_test_any = cute.ffi(name="nvshmem_size_test_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ptrdiff_test_any = cute.ffi(name="nvshmem_ptrdiff_test_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

short_test_some = cute.ffi(name="nvshmem_short_test_some", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16], return_type=cutlass.Uint64)

int_test_some = cute.ffi(name="nvshmem_int_test_some", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

long_test_some = cute.ffi(name="nvshmem_long_test_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

longlong_test_some = cute.ffi(name="nvshmem_longlong_test_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

ushort_test_some = cute.ffi(name="nvshmem_ushort_test_some", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Uint64)

uint_test_some = cute.ffi(name="nvshmem_uint_test_some", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

ulong_test_some = cute.ffi(name="nvshmem_ulong_test_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ulonglong_test_some = cute.ffi(name="nvshmem_ulonglong_test_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

int32_test_some = cute.ffi(name="nvshmem_int32_test_some", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

int64_test_some = cute.ffi(name="nvshmem_int64_test_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

uint32_test_some = cute.ffi(name="nvshmem_uint32_test_some", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

uint64_test_some = cute.ffi(name="nvshmem_uint64_test_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

size_test_some = cute.ffi(name="nvshmem_size_test_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ptrdiff_test_some = cute.ffi(name="nvshmem_ptrdiff_test_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

short_test_all_vector = cute.ffi(name="nvshmem_short_test_all_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)], return_type=cutlass.Int32)

int_test_all_vector = cute.ffi(name="nvshmem_int_test_all_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Int32)

long_test_all_vector = cute.ffi(name="nvshmem_long_test_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Int32)

longlong_test_all_vector = cute.ffi(name="nvshmem_longlong_test_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Int32)

ushort_test_all_vector = cute.ffi(name="nvshmem_ushort_test_all_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)], return_type=cutlass.Int32)

uint_test_all_vector = cute.ffi(name="nvshmem_uint_test_all_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Int32)

ulong_test_all_vector = cute.ffi(name="nvshmem_ulong_test_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Int32)

ulonglong_test_all_vector = cute.ffi(name="nvshmem_ulonglong_test_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Int32)

int32_test_all_vector = cute.ffi(name="nvshmem_int32_test_all_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Int32)

int64_test_all_vector = cute.ffi(name="nvshmem_int64_test_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Int32)

uint32_test_all_vector = cute.ffi(name="nvshmem_uint32_test_all_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Int32)

uint64_test_all_vector = cute.ffi(name="nvshmem_uint64_test_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Int32)

size_test_all_vector = cute.ffi(name="nvshmem_size_test_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Int32)

ptrdiff_test_all_vector = cute.ffi(name="nvshmem_ptrdiff_test_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Int32)

short_test_any_vector = cute.ffi(name="nvshmem_short_test_any_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)], return_type=cutlass.Uint64)

int_test_any_vector = cute.ffi(name="nvshmem_int_test_any_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

long_test_any_vector = cute.ffi(name="nvshmem_long_test_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

longlong_test_any_vector = cute.ffi(name="nvshmem_longlong_test_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

ushort_test_any_vector = cute.ffi(name="nvshmem_ushort_test_any_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)], return_type=cutlass.Uint64)

uint_test_any_vector = cute.ffi(name="nvshmem_uint_test_any_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

ulong_test_any_vector = cute.ffi(name="nvshmem_ulong_test_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ulonglong_test_any_vector = cute.ffi(name="nvshmem_ulonglong_test_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

int32_test_any_vector = cute.ffi(name="nvshmem_int32_test_any_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

int64_test_any_vector = cute.ffi(name="nvshmem_int64_test_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

uint32_test_any_vector = cute.ffi(name="nvshmem_uint32_test_any_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

uint64_test_any_vector = cute.ffi(name="nvshmem_uint64_test_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

size_test_any_vector = cute.ffi(name="nvshmem_size_test_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ptrdiff_test_any_vector = cute.ffi(name="nvshmem_ptrdiff_test_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

short_test_some_vector = cute.ffi(name="nvshmem_short_test_some_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)], return_type=cutlass.Uint64)

int_test_some_vector = cute.ffi(name="nvshmem_int_test_some_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

long_test_some_vector = cute.ffi(name="nvshmem_long_test_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

longlong_test_some_vector = cute.ffi(name="nvshmem_longlong_test_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

ushort_test_some_vector = cute.ffi(name="nvshmem_ushort_test_some_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)], return_type=cutlass.Uint64)

uint_test_some_vector = cute.ffi(name="nvshmem_uint_test_some_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

ulong_test_some_vector = cute.ffi(name="nvshmem_ulong_test_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ulonglong_test_some_vector = cute.ffi(name="nvshmem_ulonglong_test_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

int32_test_some_vector = cute.ffi(name="nvshmem_int32_test_some_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

int64_test_some_vector = cute.ffi(name="nvshmem_int64_test_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

uint32_test_some_vector = cute.ffi(name="nvshmem_uint32_test_some_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

uint64_test_some_vector = cute.ffi(name="nvshmem_uint64_test_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

size_test_some_vector = cute.ffi(name="nvshmem_size_test_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ptrdiff_test_some_vector = cute.ffi(name="nvshmem_ptrdiff_test_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

short_wait_until = cute.ffi(name="nvshmem_short_wait_until", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32, cutlass.Int16])

int_wait_until = cute.ffi(name="nvshmem_int_wait_until", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

long_wait_until = cute.ffi(name="nvshmem_long_wait_until", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64])

longlong_wait_until = cute.ffi(name="nvshmem_longlong_wait_until", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64])

ushort_wait_until = cute.ffi(name="nvshmem_ushort_wait_until", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32, cutlass.Uint16])

uint_wait_until = cute.ffi(name="nvshmem_uint_wait_until", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Uint32])

ulong_wait_until = cute.ffi(name="nvshmem_ulong_wait_until", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64])

ulonglong_wait_until = cute.ffi(name="nvshmem_ulonglong_wait_until", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64])

int32_wait_until = cute.ffi(name="nvshmem_int32_wait_until", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_wait_until = cute.ffi(name="nvshmem_int64_wait_until", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64])

uint32_wait_until = cute.ffi(name="nvshmem_uint32_wait_until", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Uint32])

uint64_wait_until = cute.ffi(name="nvshmem_uint64_wait_until", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64])

size_wait_until = cute.ffi(name="nvshmem_size_wait_until", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64])

ptrdiff_wait_until = cute.ffi(name="nvshmem_ptrdiff_wait_until", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int64])

signal_fetch = cute.ffi(name="nvshmem_signal_fetch", params_types=[_CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

signal_wait_until = cute.ffi(name="nvshmem_signal_wait_until", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

short_wait_until_all = cute.ffi(name="nvshmem_short_wait_until_all", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16])

int_wait_until_all = cute.ffi(name="nvshmem_int_wait_until_all", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

long_wait_until_all = cute.ffi(name="nvshmem_long_wait_until_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64])

longlong_wait_until_all = cute.ffi(name="nvshmem_longlong_wait_until_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64])

ushort_wait_until_all = cute.ffi(name="nvshmem_ushort_wait_until_all", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16])

uint_wait_until_all = cute.ffi(name="nvshmem_uint_wait_until_all", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32])

ulong_wait_until_all = cute.ffi(name="nvshmem_ulong_wait_until_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64])

ulonglong_wait_until_all = cute.ffi(name="nvshmem_ulonglong_wait_until_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64])

int32_wait_until_all = cute.ffi(name="nvshmem_int32_wait_until_all", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_wait_until_all = cute.ffi(name="nvshmem_int64_wait_until_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64])

uint32_wait_until_all = cute.ffi(name="nvshmem_uint32_wait_until_all", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32])

uint64_wait_until_all = cute.ffi(name="nvshmem_uint64_wait_until_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64])

size_wait_until_all = cute.ffi(name="nvshmem_size_wait_until_all", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64])

ptrdiff_wait_until_all = cute.ffi(name="nvshmem_ptrdiff_wait_until_all", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64])

short_wait_until_any = cute.ffi(name="nvshmem_short_wait_until_any", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16], return_type=cutlass.Uint64)

int_wait_until_any = cute.ffi(name="nvshmem_int_wait_until_any", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

long_wait_until_any = cute.ffi(name="nvshmem_long_wait_until_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

longlong_wait_until_any = cute.ffi(name="nvshmem_longlong_wait_until_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

ushort_wait_until_any = cute.ffi(name="nvshmem_ushort_wait_until_any", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Uint64)

uint_wait_until_any = cute.ffi(name="nvshmem_uint_wait_until_any", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

ulong_wait_until_any = cute.ffi(name="nvshmem_ulong_wait_until_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ulonglong_wait_until_any = cute.ffi(name="nvshmem_ulonglong_wait_until_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

int32_wait_until_any = cute.ffi(name="nvshmem_int32_wait_until_any", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

int64_wait_until_any = cute.ffi(name="nvshmem_int64_wait_until_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

uint32_wait_until_any = cute.ffi(name="nvshmem_uint32_wait_until_any", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

uint64_wait_until_any = cute.ffi(name="nvshmem_uint64_wait_until_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

size_wait_until_any = cute.ffi(name="nvshmem_size_wait_until_any", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ptrdiff_wait_until_any = cute.ffi(name="nvshmem_ptrdiff_wait_until_any", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

short_wait_until_some = cute.ffi(name="nvshmem_short_wait_until_some", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int16], return_type=cutlass.Uint64)

int_wait_until_some = cute.ffi(name="nvshmem_int_wait_until_some", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

long_wait_until_some = cute.ffi(name="nvshmem_long_wait_until_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

longlong_wait_until_some = cute.ffi(name="nvshmem_longlong_wait_until_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

ushort_wait_until_some = cute.ffi(name="nvshmem_ushort_wait_until_some", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint16], return_type=cutlass.Uint64)

uint_wait_until_some = cute.ffi(name="nvshmem_uint_wait_until_some", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

ulong_wait_until_some = cute.ffi(name="nvshmem_ulong_wait_until_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ulonglong_wait_until_some = cute.ffi(name="nvshmem_ulonglong_wait_until_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

int32_wait_until_some = cute.ffi(name="nvshmem_int32_wait_until_some", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

int64_wait_until_some = cute.ffi(name="nvshmem_int64_wait_until_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

uint32_wait_until_some = cute.ffi(name="nvshmem_uint32_wait_until_some", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint32], return_type=cutlass.Uint64)

uint64_wait_until_some = cute.ffi(name="nvshmem_uint64_wait_until_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

size_wait_until_some = cute.ffi(name="nvshmem_size_wait_until_some", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Uint64], return_type=cutlass.Uint64)

ptrdiff_wait_until_some = cute.ffi(name="nvshmem_ptrdiff_wait_until_some", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int64], return_type=cutlass.Uint64)

short_wait_until_all_vector = cute.ffi(name="nvshmem_short_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)])

int_wait_until_all_vector = cute.ffi(name="nvshmem_int_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)])

long_wait_until_all_vector = cute.ffi(name="nvshmem_long_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)])

longlong_wait_until_all_vector = cute.ffi(name="nvshmem_longlong_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)])

ushort_wait_until_all_vector = cute.ffi(name="nvshmem_ushort_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)])

uint_wait_until_all_vector = cute.ffi(name="nvshmem_uint_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)])

ulong_wait_until_all_vector = cute.ffi(name="nvshmem_ulong_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)])

ulonglong_wait_until_all_vector = cute.ffi(name="nvshmem_ulonglong_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)])

int32_wait_until_all_vector = cute.ffi(name="nvshmem_int32_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)])

int64_wait_until_all_vector = cute.ffi(name="nvshmem_int64_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)])

uint32_wait_until_all_vector = cute.ffi(name="nvshmem_uint32_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)])

uint64_wait_until_all_vector = cute.ffi(name="nvshmem_uint64_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)])

size_wait_until_all_vector = cute.ffi(name="nvshmem_size_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)])

ptrdiff_wait_until_all_vector = cute.ffi(name="nvshmem_ptrdiff_wait_until_all_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)])

short_wait_until_any_vector = cute.ffi(name="nvshmem_short_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)], return_type=cutlass.Uint64)

int_wait_until_any_vector = cute.ffi(name="nvshmem_int_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

long_wait_until_any_vector = cute.ffi(name="nvshmem_long_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

longlong_wait_until_any_vector = cute.ffi(name="nvshmem_longlong_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

ushort_wait_until_any_vector = cute.ffi(name="nvshmem_ushort_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)], return_type=cutlass.Uint64)

uint_wait_until_any_vector = cute.ffi(name="nvshmem_uint_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

ulong_wait_until_any_vector = cute.ffi(name="nvshmem_ulong_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ulonglong_wait_until_any_vector = cute.ffi(name="nvshmem_ulonglong_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

int32_wait_until_any_vector = cute.ffi(name="nvshmem_int32_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

int64_wait_until_any_vector = cute.ffi(name="nvshmem_int64_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

uint32_wait_until_any_vector = cute.ffi(name="nvshmem_uint32_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

uint64_wait_until_any_vector = cute.ffi(name="nvshmem_uint64_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

size_wait_until_any_vector = cute.ffi(name="nvshmem_size_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ptrdiff_wait_until_any_vector = cute.ffi(name="nvshmem_ptrdiff_wait_until_any_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

short_wait_until_some_vector = cute.ffi(name="nvshmem_short_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int16)], return_type=cutlass.Uint64)

int_wait_until_some_vector = cute.ffi(name="nvshmem_int_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

long_wait_until_some_vector = cute.ffi(name="nvshmem_long_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

longlong_wait_until_some_vector = cute.ffi(name="nvshmem_longlong_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

ushort_wait_until_some_vector = cute.ffi(name="nvshmem_ushort_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint16)], return_type=cutlass.Uint64)

uint_wait_until_some_vector = cute.ffi(name="nvshmem_uint_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

ulong_wait_until_some_vector = cute.ffi(name="nvshmem_ulong_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ulonglong_wait_until_some_vector = cute.ffi(name="nvshmem_ulonglong_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

int32_wait_until_some_vector = cute.ffi(name="nvshmem_int32_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int32)], return_type=cutlass.Uint64)

int64_wait_until_some_vector = cute.ffi(name="nvshmem_int64_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

uint32_wait_until_some_vector = cute.ffi(name="nvshmem_uint32_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint32)], return_type=cutlass.Uint64)

uint64_wait_until_some_vector = cute.ffi(name="nvshmem_uint64_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

size_wait_until_some_vector = cute.ffi(name="nvshmem_size_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Uint64)], return_type=cutlass.Uint64)

ptrdiff_wait_until_some_vector = cute.ffi(name="nvshmem_ptrdiff_wait_until_some_vector", params_types=[_CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Int32), cutlass.Int32, _CutePtrType(cutlass.Int64)], return_type=cutlass.Uint64)

quiet = cute.ffi(name="nvshmem_quiet", params_types=[])

fence = cute.ffi(name="nvshmem_fence", params_types=[])

int_atomic_fetch_add = cute.ffi(name="nvshmem_int_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

long_atomic_fetch_add = cute.ffi(name="nvshmem_long_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_fetch_add = cute.ffi(name="nvshmem_uint_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch_add = cute.ffi(name="nvshmem_ulong_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch_add = cute.ffi(name="nvshmem_ulonglong_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch_add = cute.ffi(name="nvshmem_int32_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

uint32_atomic_fetch_add = cute.ffi(name="nvshmem_uint32_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch_add = cute.ffi(name="nvshmem_uint64_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

size_atomic_fetch_add = cute.ffi(name="nvshmem_size_atomic_fetch_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int_atomic_add = cute.ffi(name="nvshmem_int_atomic_add", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

long_atomic_add = cute.ffi(name="nvshmem_long_atomic_add", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint_atomic_add = cute.ffi(name="nvshmem_uint_atomic_add", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_atomic_add = cute.ffi(name="nvshmem_ulong_atomic_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_atomic_add = cute.ffi(name="nvshmem_ulonglong_atomic_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int32_atomic_add = cute.ffi(name="nvshmem_int32_atomic_add", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

uint32_atomic_add = cute.ffi(name="nvshmem_uint32_atomic_add", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_atomic_add = cute.ffi(name="nvshmem_uint64_atomic_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_atomic_add = cute.ffi(name="nvshmem_size_atomic_add", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int_atomic_fetch_inc = cute.ffi(name="nvshmem_int_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

long_atomic_fetch_inc = cute.ffi(name="nvshmem_long_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_fetch_inc = cute.ffi(name="nvshmem_uint_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch_inc = cute.ffi(name="nvshmem_ulong_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch_inc = cute.ffi(name="nvshmem_ulonglong_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch_inc = cute.ffi(name="nvshmem_int32_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

uint32_atomic_fetch_inc = cute.ffi(name="nvshmem_uint32_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch_inc = cute.ffi(name="nvshmem_uint64_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

size_atomic_fetch_inc = cute.ffi(name="nvshmem_size_atomic_fetch_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

int_atomic_inc = cute.ffi(name="nvshmem_int_atomic_inc", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32])

long_atomic_inc = cute.ffi(name="nvshmem_long_atomic_inc", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32])

uint_atomic_inc = cute.ffi(name="nvshmem_uint_atomic_inc", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32])

ulong_atomic_inc = cute.ffi(name="nvshmem_ulong_atomic_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32])

ulonglong_atomic_inc = cute.ffi(name="nvshmem_ulonglong_atomic_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32])

int32_atomic_inc = cute.ffi(name="nvshmem_int32_atomic_inc", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32])

uint32_atomic_inc = cute.ffi(name="nvshmem_uint32_atomic_inc", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32])

uint64_atomic_inc = cute.ffi(name="nvshmem_uint64_atomic_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32])

size_atomic_inc = cute.ffi(name="nvshmem_size_atomic_inc", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32])

int_atomic_compare_swap = cute.ffi(name="nvshmem_int_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

long_atomic_compare_swap = cute.ffi(name="nvshmem_long_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

longlong_atomic_compare_swap = cute.ffi(name="nvshmem_longlong_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_compare_swap = cute.ffi(name="nvshmem_uint_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_compare_swap = cute.ffi(name="nvshmem_ulong_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_compare_swap = cute.ffi(name="nvshmem_ulonglong_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_compare_swap = cute.ffi(name="nvshmem_int32_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_compare_swap = cute.ffi(name="nvshmem_int64_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_compare_swap = cute.ffi(name="nvshmem_uint32_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_compare_swap = cute.ffi(name="nvshmem_uint64_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

size_atomic_compare_swap = cute.ffi(name="nvshmem_size_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ptrdiff_atomic_compare_swap = cute.ffi(name="nvshmem_ptrdiff_atomic_compare_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_fetch_and = cute.ffi(name="nvshmem_uint_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch_and = cute.ffi(name="nvshmem_ulong_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch_and = cute.ffi(name="nvshmem_ulonglong_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch_and = cute.ffi(name="nvshmem_int32_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_fetch_and = cute.ffi(name="nvshmem_int64_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_fetch_and = cute.ffi(name="nvshmem_uint32_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch_and = cute.ffi(name="nvshmem_uint64_atomic_fetch_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

uint_atomic_and = cute.ffi(name="nvshmem_uint_atomic_and", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_atomic_and = cute.ffi(name="nvshmem_ulong_atomic_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_atomic_and = cute.ffi(name="nvshmem_ulonglong_atomic_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int32_atomic_and = cute.ffi(name="nvshmem_int32_atomic_and", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_atomic_and = cute.ffi(name="nvshmem_int64_atomic_and", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint32_atomic_and = cute.ffi(name="nvshmem_uint32_atomic_and", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_atomic_and = cute.ffi(name="nvshmem_uint64_atomic_and", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint_atomic_fetch_or = cute.ffi(name="nvshmem_uint_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch_or = cute.ffi(name="nvshmem_ulong_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch_or = cute.ffi(name="nvshmem_ulonglong_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch_or = cute.ffi(name="nvshmem_int32_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_fetch_or = cute.ffi(name="nvshmem_int64_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_fetch_or = cute.ffi(name="nvshmem_uint32_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch_or = cute.ffi(name="nvshmem_uint64_atomic_fetch_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

uint_atomic_or = cute.ffi(name="nvshmem_uint_atomic_or", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_atomic_or = cute.ffi(name="nvshmem_ulong_atomic_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_atomic_or = cute.ffi(name="nvshmem_ulonglong_atomic_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int32_atomic_or = cute.ffi(name="nvshmem_int32_atomic_or", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_atomic_or = cute.ffi(name="nvshmem_int64_atomic_or", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint32_atomic_or = cute.ffi(name="nvshmem_uint32_atomic_or", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_atomic_or = cute.ffi(name="nvshmem_uint64_atomic_or", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint_atomic_fetch_xor = cute.ffi(name="nvshmem_uint_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch_xor = cute.ffi(name="nvshmem_ulong_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch_xor = cute.ffi(name="nvshmem_ulonglong_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch_xor = cute.ffi(name="nvshmem_int32_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_fetch_xor = cute.ffi(name="nvshmem_int64_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_fetch_xor = cute.ffi(name="nvshmem_uint32_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch_xor = cute.ffi(name="nvshmem_uint64_atomic_fetch_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

uint_atomic_xor = cute.ffi(name="nvshmem_uint_atomic_xor", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_atomic_xor = cute.ffi(name="nvshmem_ulong_atomic_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_atomic_xor = cute.ffi(name="nvshmem_ulonglong_atomic_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int32_atomic_xor = cute.ffi(name="nvshmem_int32_atomic_xor", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_atomic_xor = cute.ffi(name="nvshmem_int64_atomic_xor", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint32_atomic_xor = cute.ffi(name="nvshmem_uint32_atomic_xor", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_atomic_xor = cute.ffi(name="nvshmem_uint64_atomic_xor", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int_atomic_swap = cute.ffi(name="nvshmem_int_atomic_swap", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

long_atomic_swap = cute.ffi(name="nvshmem_long_atomic_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

longlong_atomic_swap = cute.ffi(name="nvshmem_longlong_atomic_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_swap = cute.ffi(name="nvshmem_uint_atomic_swap", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_swap = cute.ffi(name="nvshmem_ulong_atomic_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_swap = cute.ffi(name="nvshmem_ulonglong_atomic_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_swap = cute.ffi(name="nvshmem_int32_atomic_swap", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_swap = cute.ffi(name="nvshmem_int64_atomic_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_swap = cute.ffi(name="nvshmem_uint32_atomic_swap", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_swap = cute.ffi(name="nvshmem_uint64_atomic_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

float_atomic_swap = cute.ffi(name="nvshmem_float_atomic_swap", params_types=[_CutePtrType(cutlass.Float32), cutlass.Float32, cutlass.Int32], return_type=cutlass.Float32)

double_atomic_swap = cute.ffi(name="nvshmem_double_atomic_swap", params_types=[_CutePtrType(cutlass.Float64), cutlass.Float64, cutlass.Int32], return_type=cutlass.Float64)

size_atomic_swap = cute.ffi(name="nvshmem_size_atomic_swap", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Uint64)

ptrdiff_atomic_swap = cute.ffi(name="nvshmem_ptrdiff_atomic_swap", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)

int_atomic_fetch = cute.ffi(name="nvshmem_int_atomic_fetch", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

long_atomic_fetch = cute.ffi(name="nvshmem_long_atomic_fetch", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

longlong_atomic_fetch = cute.ffi(name="nvshmem_longlong_atomic_fetch", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

uint_atomic_fetch = cute.ffi(name="nvshmem_uint_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

ulong_atomic_fetch = cute.ffi(name="nvshmem_ulong_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

ulonglong_atomic_fetch = cute.ffi(name="nvshmem_ulonglong_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

int32_atomic_fetch = cute.ffi(name="nvshmem_int32_atomic_fetch", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32], return_type=cutlass.Int32)

int64_atomic_fetch = cute.ffi(name="nvshmem_int64_atomic_fetch", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

uint32_atomic_fetch = cute.ffi(name="nvshmem_uint32_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32], return_type=cutlass.Uint32)

uint64_atomic_fetch = cute.ffi(name="nvshmem_uint64_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

float_atomic_fetch = cute.ffi(name="nvshmem_float_atomic_fetch", params_types=[_CutePtrType(cutlass.Float32), cutlass.Int32], return_type=cutlass.Float32)

double_atomic_fetch = cute.ffi(name="nvshmem_double_atomic_fetch", params_types=[_CutePtrType(cutlass.Float64), cutlass.Int32], return_type=cutlass.Float64)

size_atomic_fetch = cute.ffi(name="nvshmem_size_atomic_fetch", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32], return_type=cutlass.Uint64)

ptrdiff_atomic_fetch = cute.ffi(name="nvshmem_ptrdiff_atomic_fetch", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32], return_type=cutlass.Int64)

int_atomic_set = cute.ffi(name="nvshmem_int_atomic_set", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

long_atomic_set = cute.ffi(name="nvshmem_long_atomic_set", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

longlong_atomic_set = cute.ffi(name="nvshmem_longlong_atomic_set", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint_atomic_set = cute.ffi(name="nvshmem_uint_atomic_set", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

ulong_atomic_set = cute.ffi(name="nvshmem_ulong_atomic_set", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_atomic_set = cute.ffi(name="nvshmem_ulonglong_atomic_set", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int32_atomic_set = cute.ffi(name="nvshmem_int32_atomic_set", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32])

int64_atomic_set = cute.ffi(name="nvshmem_int64_atomic_set", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

uint32_atomic_set = cute.ffi(name="nvshmem_uint32_atomic_set", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32])

uint64_atomic_set = cute.ffi(name="nvshmem_uint64_atomic_set", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

float_atomic_set = cute.ffi(name="nvshmem_float_atomic_set", params_types=[_CutePtrType(cutlass.Float32), cutlass.Float32, cutlass.Int32])

double_atomic_set = cute.ffi(name="nvshmem_double_atomic_set", params_types=[_CutePtrType(cutlass.Float64), cutlass.Float64, cutlass.Int32])

size_atomic_set = cute.ffi(name="nvshmem_size_atomic_set", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_atomic_set = cute.ffi(name="nvshmem_ptrdiff_atomic_set", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32])

ptr = cute.ffi(name="nvshmem_ptr", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32])

alltoallmem = cute.ffi(name="nvshmem_alltoallmem", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_alltoall = cute.ffi(name="nvshmem_bfloat16_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_alltoall = cute.ffi(name="nvshmem_half_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_alltoall = cute.ffi(name="nvshmem_float_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_alltoall = cute.ffi(name="nvshmem_double_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_alltoall = cute.ffi(name="nvshmem_char_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_alltoall = cute.ffi(name="nvshmem_short_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_alltoall = cute.ffi(name="nvshmem_schar_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_alltoall = cute.ffi(name="nvshmem_int_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_alltoall = cute.ffi(name="nvshmem_long_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_alltoall = cute.ffi(name="nvshmem_longlong_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_alltoall = cute.ffi(name="nvshmem_uchar_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_alltoall = cute.ffi(name="nvshmem_ushort_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_alltoall = cute.ffi(name="nvshmem_uint_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_alltoall = cute.ffi(name="nvshmem_ulong_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_alltoall = cute.ffi(name="nvshmem_ulonglong_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_alltoall = cute.ffi(name="nvshmem_int8_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_alltoall = cute.ffi(name="nvshmem_int16_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_alltoall = cute.ffi(name="nvshmem_int32_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_alltoall = cute.ffi(name="nvshmem_int64_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_alltoall = cute.ffi(name="nvshmem_uint8_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_alltoall = cute.ffi(name="nvshmem_uint16_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_alltoall = cute.ffi(name="nvshmem_uint32_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_alltoall = cute.ffi(name="nvshmem_uint64_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_alltoall = cute.ffi(name="nvshmem_size_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_alltoall = cute.ffi(name="nvshmem_ptrdiff_alltoall", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

barrier = cute.ffi(name="nvshmem_barrier", params_types=[cutlass.Int32], return_type=cutlass.Int32)

barrier_all = cute.ffi(name="nvshmem_barrier_all", params_types=[])

team_sync = cute.ffi(name="nvshmem_team_sync", params_types=[cutlass.Int32], return_type=cutlass.Int32)

sync_all = cute.ffi(name="nvshmem_sync_all", params_types=[])

broadcastmem = cute.ffi(name="nvshmem_broadcastmem", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

bfloat16_broadcast = cute.ffi(name="nvshmem_bfloat16_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

half_broadcast = cute.ffi(name="nvshmem_half_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

float_broadcast = cute.ffi(name="nvshmem_float_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

double_broadcast = cute.ffi(name="nvshmem_double_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

char_broadcast = cute.ffi(name="nvshmem_char_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

short_broadcast = cute.ffi(name="nvshmem_short_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

schar_broadcast = cute.ffi(name="nvshmem_schar_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int_broadcast = cute.ffi(name="nvshmem_int_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

long_broadcast = cute.ffi(name="nvshmem_long_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

longlong_broadcast = cute.ffi(name="nvshmem_longlong_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uchar_broadcast = cute.ffi(name="nvshmem_uchar_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ushort_broadcast = cute.ffi(name="nvshmem_ushort_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint_broadcast = cute.ffi(name="nvshmem_uint_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulong_broadcast = cute.ffi(name="nvshmem_ulong_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulonglong_broadcast = cute.ffi(name="nvshmem_ulonglong_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int8_broadcast = cute.ffi(name="nvshmem_int8_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int16_broadcast = cute.ffi(name="nvshmem_int16_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int32_broadcast = cute.ffi(name="nvshmem_int32_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int64_broadcast = cute.ffi(name="nvshmem_int64_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint8_broadcast = cute.ffi(name="nvshmem_uint8_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint16_broadcast = cute.ffi(name="nvshmem_uint16_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint32_broadcast = cute.ffi(name="nvshmem_uint32_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint64_broadcast = cute.ffi(name="nvshmem_uint64_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

size_broadcast = cute.ffi(name="nvshmem_size_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ptrdiff_broadcast = cute.ffi(name="nvshmem_ptrdiff_broadcast", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

fcollectmem = cute.ffi(name="nvshmem_fcollectmem", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_fcollect = cute.ffi(name="nvshmem_bfloat16_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_fcollect = cute.ffi(name="nvshmem_half_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_fcollect = cute.ffi(name="nvshmem_float_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_fcollect = cute.ffi(name="nvshmem_double_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_fcollect = cute.ffi(name="nvshmem_char_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_fcollect = cute.ffi(name="nvshmem_short_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_fcollect = cute.ffi(name="nvshmem_schar_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_fcollect = cute.ffi(name="nvshmem_int_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_fcollect = cute.ffi(name="nvshmem_long_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_fcollect = cute.ffi(name="nvshmem_longlong_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_fcollect = cute.ffi(name="nvshmem_uchar_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_fcollect = cute.ffi(name="nvshmem_ushort_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_fcollect = cute.ffi(name="nvshmem_uint_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_fcollect = cute.ffi(name="nvshmem_ulong_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_fcollect = cute.ffi(name="nvshmem_ulonglong_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_fcollect = cute.ffi(name="nvshmem_int8_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_fcollect = cute.ffi(name="nvshmem_int16_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_fcollect = cute.ffi(name="nvshmem_int32_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_fcollect = cute.ffi(name="nvshmem_int64_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_fcollect = cute.ffi(name="nvshmem_uint8_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_fcollect = cute.ffi(name="nvshmem_uint16_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_fcollect = cute.ffi(name="nvshmem_uint32_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_fcollect = cute.ffi(name="nvshmem_uint64_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_fcollect = cute.ffi(name="nvshmem_size_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_fcollect = cute.ffi(name="nvshmem_ptrdiff_fcollect", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reduce = cute.ffi(name="nvshmem_uchar_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reduce = cute.ffi(name="nvshmem_ushort_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reduce = cute.ffi(name="nvshmem_uint_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reduce = cute.ffi(name="nvshmem_ulong_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reduce = cute.ffi(name="nvshmem_ulonglong_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reduce = cute.ffi(name="nvshmem_int8_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reduce = cute.ffi(name="nvshmem_int16_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reduce = cute.ffi(name="nvshmem_int32_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reduce = cute.ffi(name="nvshmem_int64_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reduce = cute.ffi(name="nvshmem_uint8_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reduce = cute.ffi(name="nvshmem_uint16_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reduce = cute.ffi(name="nvshmem_uint32_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reduce = cute.ffi(name="nvshmem_uint64_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reduce = cute.ffi(name="nvshmem_size_and_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reduce = cute.ffi(name="nvshmem_uchar_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reduce = cute.ffi(name="nvshmem_ushort_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reduce = cute.ffi(name="nvshmem_uint_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reduce = cute.ffi(name="nvshmem_ulong_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reduce = cute.ffi(name="nvshmem_ulonglong_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reduce = cute.ffi(name="nvshmem_int8_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reduce = cute.ffi(name="nvshmem_int16_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reduce = cute.ffi(name="nvshmem_int32_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reduce = cute.ffi(name="nvshmem_int64_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reduce = cute.ffi(name="nvshmem_uint8_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reduce = cute.ffi(name="nvshmem_uint16_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reduce = cute.ffi(name="nvshmem_uint32_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reduce = cute.ffi(name="nvshmem_uint64_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reduce = cute.ffi(name="nvshmem_size_or_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reduce = cute.ffi(name="nvshmem_uchar_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reduce = cute.ffi(name="nvshmem_ushort_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reduce = cute.ffi(name="nvshmem_uint_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reduce = cute.ffi(name="nvshmem_ulong_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reduce = cute.ffi(name="nvshmem_ulonglong_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reduce = cute.ffi(name="nvshmem_int8_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reduce = cute.ffi(name="nvshmem_int16_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reduce = cute.ffi(name="nvshmem_int32_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reduce = cute.ffi(name="nvshmem_int64_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reduce = cute.ffi(name="nvshmem_uint8_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reduce = cute.ffi(name="nvshmem_uint16_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reduce = cute.ffi(name="nvshmem_uint32_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reduce = cute.ffi(name="nvshmem_uint64_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reduce = cute.ffi(name="nvshmem_size_xor_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reduce = cute.ffi(name="nvshmem_uchar_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reduce = cute.ffi(name="nvshmem_ushort_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reduce = cute.ffi(name="nvshmem_uint_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reduce = cute.ffi(name="nvshmem_ulong_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reduce = cute.ffi(name="nvshmem_ulonglong_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reduce = cute.ffi(name="nvshmem_int8_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reduce = cute.ffi(name="nvshmem_int16_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reduce = cute.ffi(name="nvshmem_int32_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reduce = cute.ffi(name="nvshmem_int64_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reduce = cute.ffi(name="nvshmem_uint8_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reduce = cute.ffi(name="nvshmem_uint16_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reduce = cute.ffi(name="nvshmem_uint32_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reduce = cute.ffi(name="nvshmem_uint64_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reduce = cute.ffi(name="nvshmem_size_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reduce = cute.ffi(name="nvshmem_char_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reduce = cute.ffi(name="nvshmem_schar_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reduce = cute.ffi(name="nvshmem_short_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reduce = cute.ffi(name="nvshmem_int_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reduce = cute.ffi(name="nvshmem_long_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reduce = cute.ffi(name="nvshmem_longlong_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reduce = cute.ffi(name="nvshmem_bfloat16_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reduce = cute.ffi(name="nvshmem_half_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reduce = cute.ffi(name="nvshmem_float_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reduce = cute.ffi(name="nvshmem_double_max_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reduce = cute.ffi(name="nvshmem_uchar_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reduce = cute.ffi(name="nvshmem_ushort_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reduce = cute.ffi(name="nvshmem_uint_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reduce = cute.ffi(name="nvshmem_ulong_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reduce = cute.ffi(name="nvshmem_ulonglong_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reduce = cute.ffi(name="nvshmem_int8_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reduce = cute.ffi(name="nvshmem_int16_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reduce = cute.ffi(name="nvshmem_int32_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reduce = cute.ffi(name="nvshmem_int64_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reduce = cute.ffi(name="nvshmem_uint8_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reduce = cute.ffi(name="nvshmem_uint16_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reduce = cute.ffi(name="nvshmem_uint32_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reduce = cute.ffi(name="nvshmem_uint64_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reduce = cute.ffi(name="nvshmem_size_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reduce = cute.ffi(name="nvshmem_char_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reduce = cute.ffi(name="nvshmem_schar_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reduce = cute.ffi(name="nvshmem_short_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reduce = cute.ffi(name="nvshmem_int_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reduce = cute.ffi(name="nvshmem_long_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reduce = cute.ffi(name="nvshmem_longlong_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reduce = cute.ffi(name="nvshmem_bfloat16_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reduce = cute.ffi(name="nvshmem_half_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reduce = cute.ffi(name="nvshmem_float_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reduce = cute.ffi(name="nvshmem_double_min_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reduce = cute.ffi(name="nvshmem_uchar_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reduce = cute.ffi(name="nvshmem_ushort_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reduce = cute.ffi(name="nvshmem_uint_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reduce = cute.ffi(name="nvshmem_ulong_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reduce = cute.ffi(name="nvshmem_ulonglong_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reduce = cute.ffi(name="nvshmem_int8_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reduce = cute.ffi(name="nvshmem_int16_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reduce = cute.ffi(name="nvshmem_int32_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reduce = cute.ffi(name="nvshmem_int64_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reduce = cute.ffi(name="nvshmem_uint8_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reduce = cute.ffi(name="nvshmem_uint16_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reduce = cute.ffi(name="nvshmem_uint32_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reduce = cute.ffi(name="nvshmem_uint64_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reduce = cute.ffi(name="nvshmem_size_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reduce = cute.ffi(name="nvshmem_char_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reduce = cute.ffi(name="nvshmem_schar_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reduce = cute.ffi(name="nvshmem_short_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reduce = cute.ffi(name="nvshmem_int_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reduce = cute.ffi(name="nvshmem_long_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reduce = cute.ffi(name="nvshmem_longlong_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reduce = cute.ffi(name="nvshmem_bfloat16_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reduce = cute.ffi(name="nvshmem_half_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reduce = cute.ffi(name="nvshmem_float_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reduce = cute.ffi(name="nvshmem_double_sum_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reduce = cute.ffi(name="nvshmem_uchar_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reduce = cute.ffi(name="nvshmem_ushort_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reduce = cute.ffi(name="nvshmem_uint_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reduce = cute.ffi(name="nvshmem_ulong_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reduce = cute.ffi(name="nvshmem_ulonglong_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reduce = cute.ffi(name="nvshmem_int8_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reduce = cute.ffi(name="nvshmem_int16_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reduce = cute.ffi(name="nvshmem_int32_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reduce = cute.ffi(name="nvshmem_int64_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reduce = cute.ffi(name="nvshmem_uint8_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reduce = cute.ffi(name="nvshmem_uint16_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reduce = cute.ffi(name="nvshmem_uint32_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reduce = cute.ffi(name="nvshmem_uint64_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reduce = cute.ffi(name="nvshmem_size_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reduce = cute.ffi(name="nvshmem_char_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reduce = cute.ffi(name="nvshmem_schar_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reduce = cute.ffi(name="nvshmem_short_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reduce = cute.ffi(name="nvshmem_int_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reduce = cute.ffi(name="nvshmem_long_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reduce = cute.ffi(name="nvshmem_longlong_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reduce = cute.ffi(name="nvshmem_bfloat16_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reduce = cute.ffi(name="nvshmem_half_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reduce = cute.ffi(name="nvshmem_float_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reduce = cute.ffi(name="nvshmem_double_prod_reduce", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reducescatter = cute.ffi(name="nvshmem_uchar_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reducescatter = cute.ffi(name="nvshmem_ushort_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reducescatter = cute.ffi(name="nvshmem_uint_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reducescatter = cute.ffi(name="nvshmem_ulong_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reducescatter = cute.ffi(name="nvshmem_ulonglong_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reducescatter = cute.ffi(name="nvshmem_int8_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reducescatter = cute.ffi(name="nvshmem_int16_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reducescatter = cute.ffi(name="nvshmem_int32_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reducescatter = cute.ffi(name="nvshmem_int64_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reducescatter = cute.ffi(name="nvshmem_uint8_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reducescatter = cute.ffi(name="nvshmem_uint16_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reducescatter = cute.ffi(name="nvshmem_uint32_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reducescatter = cute.ffi(name="nvshmem_uint64_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reducescatter = cute.ffi(name="nvshmem_size_and_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reducescatter = cute.ffi(name="nvshmem_uchar_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reducescatter = cute.ffi(name="nvshmem_ushort_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reducescatter = cute.ffi(name="nvshmem_uint_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reducescatter = cute.ffi(name="nvshmem_ulong_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reducescatter = cute.ffi(name="nvshmem_ulonglong_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reducescatter = cute.ffi(name="nvshmem_int8_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reducescatter = cute.ffi(name="nvshmem_int16_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reducescatter = cute.ffi(name="nvshmem_int32_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reducescatter = cute.ffi(name="nvshmem_int64_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reducescatter = cute.ffi(name="nvshmem_uint8_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reducescatter = cute.ffi(name="nvshmem_uint16_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reducescatter = cute.ffi(name="nvshmem_uint32_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reducescatter = cute.ffi(name="nvshmem_uint64_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reducescatter = cute.ffi(name="nvshmem_size_or_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reducescatter = cute.ffi(name="nvshmem_uchar_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reducescatter = cute.ffi(name="nvshmem_ushort_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reducescatter = cute.ffi(name="nvshmem_uint_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reducescatter = cute.ffi(name="nvshmem_ulong_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reducescatter = cute.ffi(name="nvshmem_ulonglong_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reducescatter = cute.ffi(name="nvshmem_int8_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reducescatter = cute.ffi(name="nvshmem_int16_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reducescatter = cute.ffi(name="nvshmem_int32_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reducescatter = cute.ffi(name="nvshmem_int64_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reducescatter = cute.ffi(name="nvshmem_uint8_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reducescatter = cute.ffi(name="nvshmem_uint16_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reducescatter = cute.ffi(name="nvshmem_uint32_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reducescatter = cute.ffi(name="nvshmem_uint64_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reducescatter = cute.ffi(name="nvshmem_size_xor_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reducescatter = cute.ffi(name="nvshmem_uchar_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reducescatter = cute.ffi(name="nvshmem_ushort_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reducescatter = cute.ffi(name="nvshmem_uint_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reducescatter = cute.ffi(name="nvshmem_ulong_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reducescatter = cute.ffi(name="nvshmem_ulonglong_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reducescatter = cute.ffi(name="nvshmem_int8_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reducescatter = cute.ffi(name="nvshmem_int16_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reducescatter = cute.ffi(name="nvshmem_int32_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reducescatter = cute.ffi(name="nvshmem_int64_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reducescatter = cute.ffi(name="nvshmem_uint8_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reducescatter = cute.ffi(name="nvshmem_uint16_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reducescatter = cute.ffi(name="nvshmem_uint32_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reducescatter = cute.ffi(name="nvshmem_uint64_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reducescatter = cute.ffi(name="nvshmem_size_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reducescatter = cute.ffi(name="nvshmem_char_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reducescatter = cute.ffi(name="nvshmem_schar_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reducescatter = cute.ffi(name="nvshmem_short_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reducescatter = cute.ffi(name="nvshmem_int_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reducescatter = cute.ffi(name="nvshmem_long_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reducescatter = cute.ffi(name="nvshmem_longlong_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reducescatter = cute.ffi(name="nvshmem_bfloat16_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reducescatter = cute.ffi(name="nvshmem_half_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reducescatter = cute.ffi(name="nvshmem_float_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reducescatter = cute.ffi(name="nvshmem_double_max_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reducescatter = cute.ffi(name="nvshmem_uchar_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reducescatter = cute.ffi(name="nvshmem_ushort_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reducescatter = cute.ffi(name="nvshmem_uint_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reducescatter = cute.ffi(name="nvshmem_ulong_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reducescatter = cute.ffi(name="nvshmem_ulonglong_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reducescatter = cute.ffi(name="nvshmem_int8_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reducescatter = cute.ffi(name="nvshmem_int16_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reducescatter = cute.ffi(name="nvshmem_int32_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reducescatter = cute.ffi(name="nvshmem_int64_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reducescatter = cute.ffi(name="nvshmem_uint8_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reducescatter = cute.ffi(name="nvshmem_uint16_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reducescatter = cute.ffi(name="nvshmem_uint32_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reducescatter = cute.ffi(name="nvshmem_uint64_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reducescatter = cute.ffi(name="nvshmem_size_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reducescatter = cute.ffi(name="nvshmem_char_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reducescatter = cute.ffi(name="nvshmem_schar_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reducescatter = cute.ffi(name="nvshmem_short_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reducescatter = cute.ffi(name="nvshmem_int_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reducescatter = cute.ffi(name="nvshmem_long_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reducescatter = cute.ffi(name="nvshmem_longlong_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reducescatter = cute.ffi(name="nvshmem_bfloat16_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reducescatter = cute.ffi(name="nvshmem_half_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reducescatter = cute.ffi(name="nvshmem_float_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reducescatter = cute.ffi(name="nvshmem_double_min_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reducescatter = cute.ffi(name="nvshmem_uchar_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reducescatter = cute.ffi(name="nvshmem_ushort_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reducescatter = cute.ffi(name="nvshmem_uint_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reducescatter = cute.ffi(name="nvshmem_ulong_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reducescatter = cute.ffi(name="nvshmem_ulonglong_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reducescatter = cute.ffi(name="nvshmem_int8_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reducescatter = cute.ffi(name="nvshmem_int16_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reducescatter = cute.ffi(name="nvshmem_int32_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reducescatter = cute.ffi(name="nvshmem_int64_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reducescatter = cute.ffi(name="nvshmem_uint8_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reducescatter = cute.ffi(name="nvshmem_uint16_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reducescatter = cute.ffi(name="nvshmem_uint32_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reducescatter = cute.ffi(name="nvshmem_uint64_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reducescatter = cute.ffi(name="nvshmem_size_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reducescatter = cute.ffi(name="nvshmem_char_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reducescatter = cute.ffi(name="nvshmem_schar_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reducescatter = cute.ffi(name="nvshmem_short_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reducescatter = cute.ffi(name="nvshmem_int_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reducescatter = cute.ffi(name="nvshmem_long_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reducescatter = cute.ffi(name="nvshmem_longlong_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reducescatter = cute.ffi(name="nvshmem_bfloat16_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reducescatter = cute.ffi(name="nvshmem_half_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reducescatter = cute.ffi(name="nvshmem_float_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reducescatter = cute.ffi(name="nvshmem_double_sum_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reducescatter = cute.ffi(name="nvshmem_uchar_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reducescatter = cute.ffi(name="nvshmem_ushort_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reducescatter = cute.ffi(name="nvshmem_uint_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reducescatter = cute.ffi(name="nvshmem_ulong_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reducescatter = cute.ffi(name="nvshmem_ulonglong_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reducescatter = cute.ffi(name="nvshmem_int8_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reducescatter = cute.ffi(name="nvshmem_int16_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reducescatter = cute.ffi(name="nvshmem_int32_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reducescatter = cute.ffi(name="nvshmem_int64_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reducescatter = cute.ffi(name="nvshmem_uint8_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reducescatter = cute.ffi(name="nvshmem_uint16_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reducescatter = cute.ffi(name="nvshmem_uint32_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reducescatter = cute.ffi(name="nvshmem_uint64_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reducescatter = cute.ffi(name="nvshmem_size_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reducescatter = cute.ffi(name="nvshmem_char_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reducescatter = cute.ffi(name="nvshmem_schar_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reducescatter = cute.ffi(name="nvshmem_short_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reducescatter = cute.ffi(name="nvshmem_int_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reducescatter = cute.ffi(name="nvshmem_long_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reducescatter = cute.ffi(name="nvshmem_longlong_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reducescatter = cute.ffi(name="nvshmem_bfloat16_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reducescatter = cute.ffi(name="nvshmem_half_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reducescatter = cute.ffi(name="nvshmem_float_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reducescatter = cute.ffi(name="nvshmem_double_prod_reducescatter", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

vendor_get_version_info = cute.ffi(name="nvshmemx_vendor_get_version_info", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32)])

signal_op = cute.ffi(name="nvshmemx_signal_op", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

mc_ptr = cute.ffi(name="nvshmemx_mc_ptr", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8)])

bfloat16_put_warp = cute.ffi(name="nvshmemx_bfloat16_put_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

bfloat16_put_block = cute.ffi(name="nvshmemx_bfloat16_put_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_put_warp = cute.ffi(name="nvshmemx_half_put_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

half_put_block = cute.ffi(name="nvshmemx_half_put_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_put_warp = cute.ffi(name="nvshmemx_float_put_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

float_put_block = cute.ffi(name="nvshmemx_float_put_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_put_warp = cute.ffi(name="nvshmemx_double_put_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

double_put_block = cute.ffi(name="nvshmemx_double_put_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_put_warp = cute.ffi(name="nvshmemx_char_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

char_put_block = cute.ffi(name="nvshmemx_char_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_put_warp = cute.ffi(name="nvshmemx_short_put_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

short_put_block = cute.ffi(name="nvshmemx_short_put_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_put_warp = cute.ffi(name="nvshmemx_schar_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

schar_put_block = cute.ffi(name="nvshmemx_schar_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_put_warp = cute.ffi(name="nvshmemx_int_put_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int_put_block = cute.ffi(name="nvshmemx_int_put_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_put_warp = cute.ffi(name="nvshmemx_long_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

long_put_block = cute.ffi(name="nvshmemx_long_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put_warp = cute.ffi(name="nvshmemx_longlong_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put_block = cute.ffi(name="nvshmemx_longlong_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_put_warp = cute.ffi(name="nvshmemx_uchar_put_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uchar_put_block = cute.ffi(name="nvshmemx_uchar_put_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_put_warp = cute.ffi(name="nvshmemx_ushort_put_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

ushort_put_block = cute.ffi(name="nvshmemx_ushort_put_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_put_warp = cute.ffi(name="nvshmemx_uint_put_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint_put_block = cute.ffi(name="nvshmemx_uint_put_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_put_warp = cute.ffi(name="nvshmemx_ulong_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulong_put_block = cute.ffi(name="nvshmemx_ulong_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put_warp = cute.ffi(name="nvshmemx_ulonglong_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put_block = cute.ffi(name="nvshmemx_ulonglong_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_put_warp = cute.ffi(name="nvshmemx_int8_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int8_put_block = cute.ffi(name="nvshmemx_int8_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_put_warp = cute.ffi(name="nvshmemx_int16_put_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int16_put_block = cute.ffi(name="nvshmemx_int16_put_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_put_warp = cute.ffi(name="nvshmemx_int32_put_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int32_put_block = cute.ffi(name="nvshmemx_int32_put_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_put_warp = cute.ffi(name="nvshmemx_int64_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

int64_put_block = cute.ffi(name="nvshmemx_int64_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_put_warp = cute.ffi(name="nvshmemx_uint8_put_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint8_put_block = cute.ffi(name="nvshmemx_uint8_put_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_put_warp = cute.ffi(name="nvshmemx_uint16_put_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint16_put_block = cute.ffi(name="nvshmemx_uint16_put_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_put_warp = cute.ffi(name="nvshmemx_uint32_put_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint32_put_block = cute.ffi(name="nvshmemx_uint32_put_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_put_warp = cute.ffi(name="nvshmemx_uint64_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint64_put_block = cute.ffi(name="nvshmemx_uint64_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put_warp = cute.ffi(name="nvshmemx_size_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put_block = cute.ffi(name="nvshmemx_size_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put_warp = cute.ffi(name="nvshmemx_ptrdiff_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put_block = cute.ffi(name="nvshmemx_ptrdiff_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

nvshmemi_bfloat16_put_signal_warp = cute.ffi(name="nvshmemi_bfloat16_put_signal_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_half_put_signal_warp = cute.ffi(name="nvshmemi_half_put_signal_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_float_put_signal_warp = cute.ffi(name="nvshmemi_float_put_signal_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_double_put_signal_warp = cute.ffi(name="nvshmemi_double_put_signal_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_char_put_signal_warp = cute.ffi(name="nvshmemi_char_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_short_put_signal_warp = cute.ffi(name="nvshmemi_short_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_schar_put_signal_warp = cute.ffi(name="nvshmemi_schar_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int_put_signal_warp = cute.ffi(name="nvshmemi_int_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_long_put_signal_warp = cute.ffi(name="nvshmemi_long_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_longlong_put_signal_warp = cute.ffi(name="nvshmemi_longlong_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uchar_put_signal_warp = cute.ffi(name="nvshmemi_uchar_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ushort_put_signal_warp = cute.ffi(name="nvshmemi_ushort_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint_put_signal_warp = cute.ffi(name="nvshmemi_uint_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ulong_put_signal_warp = cute.ffi(name="nvshmemi_ulong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ulonglong_put_signal_warp = cute.ffi(name="nvshmemi_ulonglong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int8_put_signal_warp = cute.ffi(name="nvshmemi_int8_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int16_put_signal_warp = cute.ffi(name="nvshmemi_int16_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int32_put_signal_warp = cute.ffi(name="nvshmemi_int32_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int64_put_signal_warp = cute.ffi(name="nvshmemi_int64_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint8_put_signal_warp = cute.ffi(name="nvshmemi_uint8_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint16_put_signal_warp = cute.ffi(name="nvshmemi_uint16_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint32_put_signal_warp = cute.ffi(name="nvshmemi_uint32_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint64_put_signal_warp = cute.ffi(name="nvshmemi_uint64_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_size_put_signal_warp = cute.ffi(name="nvshmemi_size_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ptrdiff_put_signal_warp = cute.ffi(name="nvshmemi_ptrdiff_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_bfloat16_put_signal_block = cute.ffi(name="nvshmemi_bfloat16_put_signal_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_half_put_signal_block = cute.ffi(name="nvshmemi_half_put_signal_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_float_put_signal_block = cute.ffi(name="nvshmemi_float_put_signal_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_double_put_signal_block = cute.ffi(name="nvshmemi_double_put_signal_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_char_put_signal_block = cute.ffi(name="nvshmemi_char_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_short_put_signal_block = cute.ffi(name="nvshmemi_short_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_schar_put_signal_block = cute.ffi(name="nvshmemi_schar_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int_put_signal_block = cute.ffi(name="nvshmemi_int_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_long_put_signal_block = cute.ffi(name="nvshmemi_long_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_longlong_put_signal_block = cute.ffi(name="nvshmemi_longlong_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uchar_put_signal_block = cute.ffi(name="nvshmemi_uchar_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ushort_put_signal_block = cute.ffi(name="nvshmemi_ushort_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint_put_signal_block = cute.ffi(name="nvshmemi_uint_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ulong_put_signal_block = cute.ffi(name="nvshmemi_ulong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ulonglong_put_signal_block = cute.ffi(name="nvshmemi_ulonglong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int8_put_signal_block = cute.ffi(name="nvshmemi_int8_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int16_put_signal_block = cute.ffi(name="nvshmemi_int16_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int32_put_signal_block = cute.ffi(name="nvshmemi_int32_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_int64_put_signal_block = cute.ffi(name="nvshmemi_int64_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint8_put_signal_block = cute.ffi(name="nvshmemi_uint8_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint16_put_signal_block = cute.ffi(name="nvshmemi_uint16_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint32_put_signal_block = cute.ffi(name="nvshmemi_uint32_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_uint64_put_signal_block = cute.ffi(name="nvshmemi_uint64_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_size_put_signal_block = cute.ffi(name="nvshmemi_size_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

nvshmemi_ptrdiff_put_signal_block = cute.ffi(name="nvshmemi_ptrdiff_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Boolean])

bfloat16_put_signal_warp = cute.ffi(name="nvshmemx_bfloat16_put_signal_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal_warp = cute.ffi(name="nvshmemx_half_put_signal_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal_warp = cute.ffi(name="nvshmemx_float_put_signal_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal_warp = cute.ffi(name="nvshmemx_double_put_signal_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal_warp = cute.ffi(name="nvshmemx_char_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal_warp = cute.ffi(name="nvshmemx_short_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal_warp = cute.ffi(name="nvshmemx_schar_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal_warp = cute.ffi(name="nvshmemx_int_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal_warp = cute.ffi(name="nvshmemx_long_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal_warp = cute.ffi(name="nvshmemx_longlong_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal_warp = cute.ffi(name="nvshmemx_uchar_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal_warp = cute.ffi(name="nvshmemx_ushort_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal_warp = cute.ffi(name="nvshmemx_uint_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal_warp = cute.ffi(name="nvshmemx_ulong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal_warp = cute.ffi(name="nvshmemx_ulonglong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal_warp = cute.ffi(name="nvshmemx_int8_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal_warp = cute.ffi(name="nvshmemx_int16_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal_warp = cute.ffi(name="nvshmemx_int32_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal_warp = cute.ffi(name="nvshmemx_int64_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal_warp = cute.ffi(name="nvshmemx_uint8_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal_warp = cute.ffi(name="nvshmemx_uint16_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal_warp = cute.ffi(name="nvshmemx_uint32_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal_warp = cute.ffi(name="nvshmemx_uint64_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal_warp = cute.ffi(name="nvshmemx_size_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal_warp = cute.ffi(name="nvshmemx_ptrdiff_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_put_signal_block = cute.ffi(name="nvshmemx_bfloat16_put_signal_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal_block = cute.ffi(name="nvshmemx_half_put_signal_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal_block = cute.ffi(name="nvshmemx_float_put_signal_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal_block = cute.ffi(name="nvshmemx_double_put_signal_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal_block = cute.ffi(name="nvshmemx_char_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal_block = cute.ffi(name="nvshmemx_short_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal_block = cute.ffi(name="nvshmemx_schar_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal_block = cute.ffi(name="nvshmemx_int_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal_block = cute.ffi(name="nvshmemx_long_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal_block = cute.ffi(name="nvshmemx_longlong_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal_block = cute.ffi(name="nvshmemx_uchar_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal_block = cute.ffi(name="nvshmemx_ushort_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal_block = cute.ffi(name="nvshmemx_uint_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal_block = cute.ffi(name="nvshmemx_ulong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal_block = cute.ffi(name="nvshmemx_ulonglong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal_block = cute.ffi(name="nvshmemx_int8_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal_block = cute.ffi(name="nvshmemx_int16_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal_block = cute.ffi(name="nvshmemx_int32_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal_block = cute.ffi(name="nvshmemx_int64_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal_block = cute.ffi(name="nvshmemx_uint8_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal_block = cute.ffi(name="nvshmemx_uint16_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal_block = cute.ffi(name="nvshmemx_uint32_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal_block = cute.ffi(name="nvshmemx_uint64_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal_block = cute.ffi(name="nvshmemx_size_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal_block = cute.ffi(name="nvshmemx_ptrdiff_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

putmem_signal_warp = cute.ffi(name="nvshmemx_putmem_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

putmem_signal_block = cute.ffi(name="nvshmemx_putmem_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal_warp = cute.ffi(name="nvshmemx_put8_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put6_signal_warp = cute.ffi(name="nvshmemx_put6_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal_warp = cute.ffi(name="nvshmemx_put32_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal_warp = cute.ffi(name="nvshmemx_put64_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal_warp = cute.ffi(name="nvshmemx_put128_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal_block = cute.ffi(name="nvshmemx_put8_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put6_signal_block = cute.ffi(name="nvshmemx_put6_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal_block = cute.ffi(name="nvshmemx_put32_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal_block = cute.ffi(name="nvshmemx_put64_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal_block = cute.ffi(name="nvshmemx_put128_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_bfloat16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal_nbi_warp = cute.ffi(name="nvshmemx_half_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal_nbi_warp = cute.ffi(name="nvshmemx_float_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal_nbi_warp = cute.ffi(name="nvshmemx_double_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal_nbi_warp = cute.ffi(name="nvshmemx_char_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal_nbi_warp = cute.ffi(name="nvshmemx_short_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal_nbi_warp = cute.ffi(name="nvshmemx_schar_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal_nbi_warp = cute.ffi(name="nvshmemx_int_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal_nbi_warp = cute.ffi(name="nvshmemx_long_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_longlong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uchar_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal_nbi_warp = cute.ffi(name="nvshmemx_ushort_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uint_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_ulong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_ulonglong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal_nbi_warp = cute.ffi(name="nvshmemx_int8_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_int16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal_nbi_warp = cute.ffi(name="nvshmemx_int32_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal_nbi_warp = cute.ffi(name="nvshmemx_int64_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uint8_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uint16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uint32_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal_nbi_warp = cute.ffi(name="nvshmemx_uint64_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal_nbi_warp = cute.ffi(name="nvshmemx_size_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal_nbi_warp = cute.ffi(name="nvshmemx_ptrdiff_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_put_signal_nbi_block = cute.ffi(name="nvshmemx_bfloat16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

half_put_signal_nbi_block = cute.ffi(name="nvshmemx_half_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

float_put_signal_nbi_block = cute.ffi(name="nvshmemx_float_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

double_put_signal_nbi_block = cute.ffi(name="nvshmemx_double_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

char_put_signal_nbi_block = cute.ffi(name="nvshmemx_char_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

short_put_signal_nbi_block = cute.ffi(name="nvshmemx_short_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

schar_put_signal_nbi_block = cute.ffi(name="nvshmemx_schar_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int_put_signal_nbi_block = cute.ffi(name="nvshmemx_int_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

long_put_signal_nbi_block = cute.ffi(name="nvshmemx_long_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

longlong_put_signal_nbi_block = cute.ffi(name="nvshmemx_longlong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uchar_put_signal_nbi_block = cute.ffi(name="nvshmemx_uchar_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ushort_put_signal_nbi_block = cute.ffi(name="nvshmemx_ushort_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint_put_signal_nbi_block = cute.ffi(name="nvshmemx_uint_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulong_put_signal_nbi_block = cute.ffi(name="nvshmemx_ulong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ulonglong_put_signal_nbi_block = cute.ffi(name="nvshmemx_ulonglong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int8_put_signal_nbi_block = cute.ffi(name="nvshmemx_int8_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int16_put_signal_nbi_block = cute.ffi(name="nvshmemx_int16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int32_put_signal_nbi_block = cute.ffi(name="nvshmemx_int32_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

int64_put_signal_nbi_block = cute.ffi(name="nvshmemx_int64_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint8_put_signal_nbi_block = cute.ffi(name="nvshmemx_uint8_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint16_put_signal_nbi_block = cute.ffi(name="nvshmemx_uint16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint32_put_signal_nbi_block = cute.ffi(name="nvshmemx_uint32_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

uint64_put_signal_nbi_block = cute.ffi(name="nvshmemx_uint64_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

size_put_signal_nbi_block = cute.ffi(name="nvshmemx_size_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

ptrdiff_put_signal_nbi_block = cute.ffi(name="nvshmemx_ptrdiff_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

putmem_signal_nbi_warp = cute.ffi(name="nvshmemx_putmem_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

putmem_signal_nbi_block = cute.ffi(name="nvshmemx_putmem_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal_nbi_warp = cute.ffi(name="nvshmemx_put8_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put6_signal_nbi_warp = cute.ffi(name="nvshmemx_put6_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal_nbi_warp = cute.ffi(name="nvshmemx_put32_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal_nbi_warp = cute.ffi(name="nvshmemx_put64_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal_nbi_warp = cute.ffi(name="nvshmemx_put128_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put8_signal_nbi_block = cute.ffi(name="nvshmemx_put8_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put6_signal_nbi_block = cute.ffi(name="nvshmemx_put6_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put32_signal_nbi_block = cute.ffi(name="nvshmemx_put32_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put64_signal_nbi_block = cute.ffi(name="nvshmemx_put64_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

put128_signal_nbi_block = cute.ffi(name="nvshmemx_put128_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

bfloat16_get_warp = cute.ffi(name="nvshmemx_bfloat16_get_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

bfloat16_get_block = cute.ffi(name="nvshmemx_bfloat16_get_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_get_warp = cute.ffi(name="nvshmemx_half_get_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

half_get_block = cute.ffi(name="nvshmemx_half_get_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_get_warp = cute.ffi(name="nvshmemx_float_get_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

float_get_block = cute.ffi(name="nvshmemx_float_get_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_get_warp = cute.ffi(name="nvshmemx_double_get_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

double_get_block = cute.ffi(name="nvshmemx_double_get_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_get_warp = cute.ffi(name="nvshmemx_char_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

char_get_block = cute.ffi(name="nvshmemx_char_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_get_warp = cute.ffi(name="nvshmemx_short_get_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

short_get_block = cute.ffi(name="nvshmemx_short_get_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_get_warp = cute.ffi(name="nvshmemx_schar_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

schar_get_block = cute.ffi(name="nvshmemx_schar_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_get_warp = cute.ffi(name="nvshmemx_int_get_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int_get_block = cute.ffi(name="nvshmemx_int_get_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_get_warp = cute.ffi(name="nvshmemx_long_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

long_get_block = cute.ffi(name="nvshmemx_long_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get_warp = cute.ffi(name="nvshmemx_longlong_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get_block = cute.ffi(name="nvshmemx_longlong_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_get_warp = cute.ffi(name="nvshmemx_uchar_get_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uchar_get_block = cute.ffi(name="nvshmemx_uchar_get_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_get_warp = cute.ffi(name="nvshmemx_ushort_get_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

ushort_get_block = cute.ffi(name="nvshmemx_ushort_get_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_get_warp = cute.ffi(name="nvshmemx_uint_get_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint_get_block = cute.ffi(name="nvshmemx_uint_get_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_get_warp = cute.ffi(name="nvshmemx_ulong_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulong_get_block = cute.ffi(name="nvshmemx_ulong_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get_warp = cute.ffi(name="nvshmemx_ulonglong_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get_block = cute.ffi(name="nvshmemx_ulonglong_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_get_warp = cute.ffi(name="nvshmemx_int8_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int8_get_block = cute.ffi(name="nvshmemx_int8_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_get_warp = cute.ffi(name="nvshmemx_int16_get_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int16_get_block = cute.ffi(name="nvshmemx_int16_get_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_get_warp = cute.ffi(name="nvshmemx_int32_get_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int32_get_block = cute.ffi(name="nvshmemx_int32_get_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_get_warp = cute.ffi(name="nvshmemx_int64_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

int64_get_block = cute.ffi(name="nvshmemx_int64_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_get_warp = cute.ffi(name="nvshmemx_uint8_get_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint8_get_block = cute.ffi(name="nvshmemx_uint8_get_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_get_warp = cute.ffi(name="nvshmemx_uint16_get_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint16_get_block = cute.ffi(name="nvshmemx_uint16_get_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_get_warp = cute.ffi(name="nvshmemx_uint32_get_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint32_get_block = cute.ffi(name="nvshmemx_uint32_get_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_get_warp = cute.ffi(name="nvshmemx_uint64_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint64_get_block = cute.ffi(name="nvshmemx_uint64_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get_warp = cute.ffi(name="nvshmemx_size_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get_block = cute.ffi(name="nvshmemx_size_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get_warp = cute.ffi(name="nvshmemx_ptrdiff_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get_block = cute.ffi(name="nvshmemx_ptrdiff_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

put8_warp = cute.ffi(name="nvshmemx_put8_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put8_block = cute.ffi(name="nvshmemx_put8_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16_warp = cute.ffi(name="nvshmemx_put16_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16_block = cute.ffi(name="nvshmemx_put16_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put32_warp = cute.ffi(name="nvshmemx_put32_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put32_block = cute.ffi(name="nvshmemx_put32_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put64_warp = cute.ffi(name="nvshmemx_put64_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put64_block = cute.ffi(name="nvshmemx_put64_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put128_warp = cute.ffi(name="nvshmemx_put128_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put128_block = cute.ffi(name="nvshmemx_put128_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get8_warp = cute.ffi(name="nvshmemx_get8_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get8_block = cute.ffi(name="nvshmemx_get8_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16_warp = cute.ffi(name="nvshmemx_get16_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16_block = cute.ffi(name="nvshmemx_get16_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get32_warp = cute.ffi(name="nvshmemx_get32_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get32_block = cute.ffi(name="nvshmemx_get32_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get64_warp = cute.ffi(name="nvshmemx_get64_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get64_block = cute.ffi(name="nvshmemx_get64_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get128_warp = cute.ffi(name="nvshmemx_get128_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get128_block = cute.ffi(name="nvshmemx_get128_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_warp = cute.ffi(name="nvshmemx_putmem_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_block = cute.ffi(name="nvshmemx_putmem_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

getmem_warp = cute.ffi(name="nvshmemx_getmem_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

getmem_block = cute.ffi(name="nvshmemx_getmem_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

bfloat16_put_nbi_warp = cute.ffi(name="nvshmemx_bfloat16_put_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

bfloat16_put_nbi_block = cute.ffi(name="nvshmemx_bfloat16_put_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_put_nbi_warp = cute.ffi(name="nvshmemx_half_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

half_put_nbi_block = cute.ffi(name="nvshmemx_half_put_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_put_nbi_warp = cute.ffi(name="nvshmemx_float_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

float_put_nbi_block = cute.ffi(name="nvshmemx_float_put_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_put_nbi_warp = cute.ffi(name="nvshmemx_double_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

double_put_nbi_block = cute.ffi(name="nvshmemx_double_put_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_put_nbi_warp = cute.ffi(name="nvshmemx_char_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

char_put_nbi_block = cute.ffi(name="nvshmemx_char_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_put_nbi_warp = cute.ffi(name="nvshmemx_short_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

short_put_nbi_block = cute.ffi(name="nvshmemx_short_put_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_put_nbi_warp = cute.ffi(name="nvshmemx_schar_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

schar_put_nbi_block = cute.ffi(name="nvshmemx_schar_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_put_nbi_warp = cute.ffi(name="nvshmemx_int_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int_put_nbi_block = cute.ffi(name="nvshmemx_int_put_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_put_nbi_warp = cute.ffi(name="nvshmemx_long_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

long_put_nbi_block = cute.ffi(name="nvshmemx_long_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put_nbi_warp = cute.ffi(name="nvshmemx_longlong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_put_nbi_block = cute.ffi(name="nvshmemx_longlong_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_put_nbi_warp = cute.ffi(name="nvshmemx_uchar_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uchar_put_nbi_block = cute.ffi(name="nvshmemx_uchar_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_put_nbi_warp = cute.ffi(name="nvshmemx_ushort_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

ushort_put_nbi_block = cute.ffi(name="nvshmemx_ushort_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_put_nbi_warp = cute.ffi(name="nvshmemx_uint_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint_put_nbi_block = cute.ffi(name="nvshmemx_uint_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_put_nbi_warp = cute.ffi(name="nvshmemx_ulong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulong_put_nbi_block = cute.ffi(name="nvshmemx_ulong_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put_nbi_warp = cute.ffi(name="nvshmemx_ulonglong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_put_nbi_block = cute.ffi(name="nvshmemx_ulonglong_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_put_nbi_warp = cute.ffi(name="nvshmemx_int8_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int8_put_nbi_block = cute.ffi(name="nvshmemx_int8_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_put_nbi_warp = cute.ffi(name="nvshmemx_int16_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int16_put_nbi_block = cute.ffi(name="nvshmemx_int16_put_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_put_nbi_warp = cute.ffi(name="nvshmemx_int32_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int32_put_nbi_block = cute.ffi(name="nvshmemx_int32_put_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_put_nbi_warp = cute.ffi(name="nvshmemx_int64_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

int64_put_nbi_block = cute.ffi(name="nvshmemx_int64_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_put_nbi_warp = cute.ffi(name="nvshmemx_uint8_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint8_put_nbi_block = cute.ffi(name="nvshmemx_uint8_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_put_nbi_warp = cute.ffi(name="nvshmemx_uint16_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint16_put_nbi_block = cute.ffi(name="nvshmemx_uint16_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_put_nbi_warp = cute.ffi(name="nvshmemx_uint32_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint32_put_nbi_block = cute.ffi(name="nvshmemx_uint32_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_put_nbi_warp = cute.ffi(name="nvshmemx_uint64_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint64_put_nbi_block = cute.ffi(name="nvshmemx_uint64_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put_nbi_warp = cute.ffi(name="nvshmemx_size_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_put_nbi_block = cute.ffi(name="nvshmemx_size_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put_nbi_warp = cute.ffi(name="nvshmemx_ptrdiff_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

ptrdiff_put_nbi_block = cute.ffi(name="nvshmemx_ptrdiff_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

bfloat16_get_nbi_warp = cute.ffi(name="nvshmemx_bfloat16_get_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

bfloat16_get_nbi_block = cute.ffi(name="nvshmemx_bfloat16_get_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32])

half_get_nbi_warp = cute.ffi(name="nvshmemx_half_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

half_get_nbi_block = cute.ffi(name="nvshmemx_half_get_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32])

float_get_nbi_warp = cute.ffi(name="nvshmemx_float_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

float_get_nbi_block = cute.ffi(name="nvshmemx_float_get_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32])

double_get_nbi_warp = cute.ffi(name="nvshmemx_double_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

double_get_nbi_block = cute.ffi(name="nvshmemx_double_get_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32])

char_get_nbi_warp = cute.ffi(name="nvshmemx_char_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

char_get_nbi_block = cute.ffi(name="nvshmemx_char_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

short_get_nbi_warp = cute.ffi(name="nvshmemx_short_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

short_get_nbi_block = cute.ffi(name="nvshmemx_short_get_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

schar_get_nbi_warp = cute.ffi(name="nvshmemx_schar_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

schar_get_nbi_block = cute.ffi(name="nvshmemx_schar_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int_get_nbi_warp = cute.ffi(name="nvshmemx_int_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int_get_nbi_block = cute.ffi(name="nvshmemx_int_get_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

long_get_nbi_warp = cute.ffi(name="nvshmemx_long_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

long_get_nbi_block = cute.ffi(name="nvshmemx_long_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get_nbi_warp = cute.ffi(name="nvshmemx_longlong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

longlong_get_nbi_block = cute.ffi(name="nvshmemx_longlong_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uchar_get_nbi_warp = cute.ffi(name="nvshmemx_uchar_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uchar_get_nbi_block = cute.ffi(name="nvshmemx_uchar_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

ushort_get_nbi_warp = cute.ffi(name="nvshmemx_ushort_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

ushort_get_nbi_block = cute.ffi(name="nvshmemx_ushort_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint_get_nbi_warp = cute.ffi(name="nvshmemx_uint_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint_get_nbi_block = cute.ffi(name="nvshmemx_uint_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

ulong_get_nbi_warp = cute.ffi(name="nvshmemx_ulong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulong_get_nbi_block = cute.ffi(name="nvshmemx_ulong_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get_nbi_warp = cute.ffi(name="nvshmemx_ulonglong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ulonglong_get_nbi_block = cute.ffi(name="nvshmemx_ulonglong_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

int8_get_nbi_warp = cute.ffi(name="nvshmemx_int8_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int8_get_nbi_block = cute.ffi(name="nvshmemx_int8_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

int16_get_nbi_warp = cute.ffi(name="nvshmemx_int16_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int16_get_nbi_block = cute.ffi(name="nvshmemx_int16_get_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32])

int32_get_nbi_warp = cute.ffi(name="nvshmemx_int32_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int32_get_nbi_block = cute.ffi(name="nvshmemx_int32_get_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32])

int64_get_nbi_warp = cute.ffi(name="nvshmemx_int64_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

int64_get_nbi_block = cute.ffi(name="nvshmemx_int64_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

uint8_get_nbi_warp = cute.ffi(name="nvshmemx_uint8_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint8_get_nbi_block = cute.ffi(name="nvshmemx_uint8_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32])

uint16_get_nbi_warp = cute.ffi(name="nvshmemx_uint16_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint16_get_nbi_block = cute.ffi(name="nvshmemx_uint16_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32])

uint32_get_nbi_warp = cute.ffi(name="nvshmemx_uint32_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint32_get_nbi_block = cute.ffi(name="nvshmemx_uint32_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32])

uint64_get_nbi_warp = cute.ffi(name="nvshmemx_uint64_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

uint64_get_nbi_block = cute.ffi(name="nvshmemx_uint64_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get_nbi_warp = cute.ffi(name="nvshmemx_size_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

size_get_nbi_block = cute.ffi(name="nvshmemx_size_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get_nbi_warp = cute.ffi(name="nvshmemx_ptrdiff_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

ptrdiff_get_nbi_block = cute.ffi(name="nvshmemx_ptrdiff_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32])

put8_nbi_warp = cute.ffi(name="nvshmemx_put8_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put8_nbi_block = cute.ffi(name="nvshmemx_put8_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16_nbi_warp = cute.ffi(name="nvshmemx_put16_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put16_nbi_block = cute.ffi(name="nvshmemx_put16_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put32_nbi_warp = cute.ffi(name="nvshmemx_put32_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put32_nbi_block = cute.ffi(name="nvshmemx_put32_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put64_nbi_warp = cute.ffi(name="nvshmemx_put64_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put64_nbi_block = cute.ffi(name="nvshmemx_put64_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put128_nbi_warp = cute.ffi(name="nvshmemx_put128_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

put128_nbi_block = cute.ffi(name="nvshmemx_put128_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get8_nbi_warp = cute.ffi(name="nvshmemx_get8_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get8_nbi_block = cute.ffi(name="nvshmemx_get8_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16_nbi_warp = cute.ffi(name="nvshmemx_get16_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get16_nbi_block = cute.ffi(name="nvshmemx_get16_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get32_nbi_warp = cute.ffi(name="nvshmemx_get32_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get32_nbi_block = cute.ffi(name="nvshmemx_get32_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get64_nbi_warp = cute.ffi(name="nvshmemx_get64_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get64_nbi_block = cute.ffi(name="nvshmemx_get64_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get128_nbi_warp = cute.ffi(name="nvshmemx_get128_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

get128_nbi_block = cute.ffi(name="nvshmemx_get128_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_nbi_warp = cute.ffi(name="nvshmemx_putmem_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

putmem_nbi_block = cute.ffi(name="nvshmemx_putmem_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

getmem_nbi_warp = cute.ffi(name="nvshmemx_getmem_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

getmem_nbi_block = cute.ffi(name="nvshmemx_getmem_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32])

bfloat16_iput_warp = cute.ffi(name="nvshmemx_bfloat16_iput_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

bfloat16_iput_block = cute.ffi(name="nvshmemx_bfloat16_iput_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iput_warp = cute.ffi(name="nvshmemx_half_iput_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iput_block = cute.ffi(name="nvshmemx_half_iput_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iput_warp = cute.ffi(name="nvshmemx_float_iput_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iput_block = cute.ffi(name="nvshmemx_float_iput_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iput_warp = cute.ffi(name="nvshmemx_double_iput_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iput_block = cute.ffi(name="nvshmemx_double_iput_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iput_warp = cute.ffi(name="nvshmemx_char_iput_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iput_block = cute.ffi(name="nvshmemx_char_iput_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iput_warp = cute.ffi(name="nvshmemx_short_iput_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iput_block = cute.ffi(name="nvshmemx_short_iput_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iput_warp = cute.ffi(name="nvshmemx_schar_iput_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iput_block = cute.ffi(name="nvshmemx_schar_iput_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iput_warp = cute.ffi(name="nvshmemx_int_iput_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iput_block = cute.ffi(name="nvshmemx_int_iput_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iput_warp = cute.ffi(name="nvshmemx_long_iput_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iput_block = cute.ffi(name="nvshmemx_long_iput_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iput_warp = cute.ffi(name="nvshmemx_longlong_iput_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iput_block = cute.ffi(name="nvshmemx_longlong_iput_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iput_warp = cute.ffi(name="nvshmemx_uchar_iput_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iput_block = cute.ffi(name="nvshmemx_uchar_iput_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iput_warp = cute.ffi(name="nvshmemx_ushort_iput_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iput_block = cute.ffi(name="nvshmemx_ushort_iput_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iput_warp = cute.ffi(name="nvshmemx_uint_iput_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iput_block = cute.ffi(name="nvshmemx_uint_iput_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iput_warp = cute.ffi(name="nvshmemx_ulong_iput_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iput_block = cute.ffi(name="nvshmemx_ulong_iput_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iput_warp = cute.ffi(name="nvshmemx_ulonglong_iput_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iput_block = cute.ffi(name="nvshmemx_ulonglong_iput_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iput_warp = cute.ffi(name="nvshmemx_int8_iput_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iput_block = cute.ffi(name="nvshmemx_int8_iput_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iput_warp = cute.ffi(name="nvshmemx_int16_iput_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iput_block = cute.ffi(name="nvshmemx_int16_iput_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iput_warp = cute.ffi(name="nvshmemx_int32_iput_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iput_block = cute.ffi(name="nvshmemx_int32_iput_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iput_warp = cute.ffi(name="nvshmemx_int64_iput_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iput_block = cute.ffi(name="nvshmemx_int64_iput_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iput_warp = cute.ffi(name="nvshmemx_uint8_iput_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iput_block = cute.ffi(name="nvshmemx_uint8_iput_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iput_warp = cute.ffi(name="nvshmemx_uint16_iput_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iput_block = cute.ffi(name="nvshmemx_uint16_iput_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iput_warp = cute.ffi(name="nvshmemx_uint32_iput_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iput_block = cute.ffi(name="nvshmemx_uint32_iput_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iput_warp = cute.ffi(name="nvshmemx_uint64_iput_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iput_block = cute.ffi(name="nvshmemx_uint64_iput_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iput_warp = cute.ffi(name="nvshmemx_size_iput_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iput_block = cute.ffi(name="nvshmemx_size_iput_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iput_warp = cute.ffi(name="nvshmemx_ptrdiff_iput_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iput_block = cute.ffi(name="nvshmemx_ptrdiff_iput_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput8_warp = cute.ffi(name="nvshmemx_iput8_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput8_block = cute.ffi(name="nvshmemx_iput8_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput16_warp = cute.ffi(name="nvshmemx_iput16_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput16_block = cute.ffi(name="nvshmemx_iput16_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput32_warp = cute.ffi(name="nvshmemx_iput32_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput32_block = cute.ffi(name="nvshmemx_iput32_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput64_warp = cute.ffi(name="nvshmemx_iput64_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput64_block = cute.ffi(name="nvshmemx_iput64_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput128_warp = cute.ffi(name="nvshmemx_iput128_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iput128_block = cute.ffi(name="nvshmemx_iput128_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

bfloat16_iget_warp = cute.ffi(name="nvshmemx_bfloat16_iget_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

bfloat16_iget_block = cute.ffi(name="nvshmemx_bfloat16_iget_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iget_warp = cute.ffi(name="nvshmemx_half_iget_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

half_iget_block = cute.ffi(name="nvshmemx_half_iget_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iget_warp = cute.ffi(name="nvshmemx_float_iget_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

float_iget_block = cute.ffi(name="nvshmemx_float_iget_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iget_warp = cute.ffi(name="nvshmemx_double_iget_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

double_iget_block = cute.ffi(name="nvshmemx_double_iget_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iget_warp = cute.ffi(name="nvshmemx_char_iget_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

char_iget_block = cute.ffi(name="nvshmemx_char_iget_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iget_warp = cute.ffi(name="nvshmemx_short_iget_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

short_iget_block = cute.ffi(name="nvshmemx_short_iget_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iget_warp = cute.ffi(name="nvshmemx_schar_iget_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

schar_iget_block = cute.ffi(name="nvshmemx_schar_iget_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iget_warp = cute.ffi(name="nvshmemx_int_iget_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int_iget_block = cute.ffi(name="nvshmemx_int_iget_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iget_warp = cute.ffi(name="nvshmemx_long_iget_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

long_iget_block = cute.ffi(name="nvshmemx_long_iget_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iget_warp = cute.ffi(name="nvshmemx_longlong_iget_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

longlong_iget_block = cute.ffi(name="nvshmemx_longlong_iget_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iget_warp = cute.ffi(name="nvshmemx_uchar_iget_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uchar_iget_block = cute.ffi(name="nvshmemx_uchar_iget_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iget_warp = cute.ffi(name="nvshmemx_ushort_iget_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ushort_iget_block = cute.ffi(name="nvshmemx_ushort_iget_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iget_warp = cute.ffi(name="nvshmemx_uint_iget_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint_iget_block = cute.ffi(name="nvshmemx_uint_iget_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iget_warp = cute.ffi(name="nvshmemx_ulong_iget_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulong_iget_block = cute.ffi(name="nvshmemx_ulong_iget_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iget_warp = cute.ffi(name="nvshmemx_ulonglong_iget_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ulonglong_iget_block = cute.ffi(name="nvshmemx_ulonglong_iget_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iget_warp = cute.ffi(name="nvshmemx_int8_iget_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int8_iget_block = cute.ffi(name="nvshmemx_int8_iget_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iget_warp = cute.ffi(name="nvshmemx_int16_iget_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int16_iget_block = cute.ffi(name="nvshmemx_int16_iget_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iget_warp = cute.ffi(name="nvshmemx_int32_iget_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int32_iget_block = cute.ffi(name="nvshmemx_int32_iget_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iget_warp = cute.ffi(name="nvshmemx_int64_iget_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

int64_iget_block = cute.ffi(name="nvshmemx_int64_iget_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iget_warp = cute.ffi(name="nvshmemx_uint8_iget_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint8_iget_block = cute.ffi(name="nvshmemx_uint8_iget_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iget_warp = cute.ffi(name="nvshmemx_uint16_iget_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint16_iget_block = cute.ffi(name="nvshmemx_uint16_iget_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iget_warp = cute.ffi(name="nvshmemx_uint32_iget_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint32_iget_block = cute.ffi(name="nvshmemx_uint32_iget_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iget_warp = cute.ffi(name="nvshmemx_uint64_iget_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

uint64_iget_block = cute.ffi(name="nvshmemx_uint64_iget_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iget_warp = cute.ffi(name="nvshmemx_size_iget_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

size_iget_block = cute.ffi(name="nvshmemx_size_iget_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iget_warp = cute.ffi(name="nvshmemx_ptrdiff_iget_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

ptrdiff_iget_block = cute.ffi(name="nvshmemx_ptrdiff_iget_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget8_warp = cute.ffi(name="nvshmemx_iget8_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget8_block = cute.ffi(name="nvshmemx_iget8_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget16_warp = cute.ffi(name="nvshmemx_iget16_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget16_block = cute.ffi(name="nvshmemx_iget16_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget32_warp = cute.ffi(name="nvshmemx_iget32_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget32_block = cute.ffi(name="nvshmemx_iget32_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget64_warp = cute.ffi(name="nvshmemx_iget64_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget64_block = cute.ffi(name="nvshmemx_iget64_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget128_warp = cute.ffi(name="nvshmemx_iget128_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

iget128_block = cute.ffi(name="nvshmemx_iget128_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Int64, cutlass.Int64, cutlass.Uint64, cutlass.Int32])

qp_bfloat16_get = cute.ffi(name="nvshmemx_qp_bfloat16_get", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_get_warp = cute.ffi(name="nvshmemx_qp_bfloat16_get_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_get_block = cute.ffi(name="nvshmemx_qp_bfloat16_get_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get = cute.ffi(name="nvshmemx_qp_half_get", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get_warp = cute.ffi(name="nvshmemx_qp_half_get_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get_block = cute.ffi(name="nvshmemx_qp_half_get_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get = cute.ffi(name="nvshmemx_qp_float_get", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get_warp = cute.ffi(name="nvshmemx_qp_float_get_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get_block = cute.ffi(name="nvshmemx_qp_float_get_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get = cute.ffi(name="nvshmemx_qp_double_get", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get_warp = cute.ffi(name="nvshmemx_qp_double_get_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get_block = cute.ffi(name="nvshmemx_qp_double_get_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get = cute.ffi(name="nvshmemx_qp_char_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get_warp = cute.ffi(name="nvshmemx_qp_char_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get_block = cute.ffi(name="nvshmemx_qp_char_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get = cute.ffi(name="nvshmemx_qp_short_get", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get_warp = cute.ffi(name="nvshmemx_qp_short_get_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get_block = cute.ffi(name="nvshmemx_qp_short_get_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get = cute.ffi(name="nvshmemx_qp_schar_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get_warp = cute.ffi(name="nvshmemx_qp_schar_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get_block = cute.ffi(name="nvshmemx_qp_schar_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get = cute.ffi(name="nvshmemx_qp_int_get", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get_warp = cute.ffi(name="nvshmemx_qp_int_get_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get_block = cute.ffi(name="nvshmemx_qp_int_get_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get = cute.ffi(name="nvshmemx_qp_long_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get_warp = cute.ffi(name="nvshmemx_qp_long_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get_block = cute.ffi(name="nvshmemx_qp_long_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get = cute.ffi(name="nvshmemx_qp_longlong_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get_warp = cute.ffi(name="nvshmemx_qp_longlong_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get_block = cute.ffi(name="nvshmemx_qp_longlong_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get = cute.ffi(name="nvshmemx_qp_uchar_get", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get_warp = cute.ffi(name="nvshmemx_qp_uchar_get_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get_block = cute.ffi(name="nvshmemx_qp_uchar_get_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get = cute.ffi(name="nvshmemx_qp_ushort_get", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get_warp = cute.ffi(name="nvshmemx_qp_ushort_get_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get_block = cute.ffi(name="nvshmemx_qp_ushort_get_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get = cute.ffi(name="nvshmemx_qp_uint_get", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get_warp = cute.ffi(name="nvshmemx_qp_uint_get_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get_block = cute.ffi(name="nvshmemx_qp_uint_get_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get = cute.ffi(name="nvshmemx_qp_ulong_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get_warp = cute.ffi(name="nvshmemx_qp_ulong_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get_block = cute.ffi(name="nvshmemx_qp_ulong_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get = cute.ffi(name="nvshmemx_qp_ulonglong_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get_warp = cute.ffi(name="nvshmemx_qp_ulonglong_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get_block = cute.ffi(name="nvshmemx_qp_ulonglong_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get = cute.ffi(name="nvshmemx_qp_int8_get", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get_warp = cute.ffi(name="nvshmemx_qp_int8_get_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get_block = cute.ffi(name="nvshmemx_qp_int8_get_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get = cute.ffi(name="nvshmemx_qp_int16_get", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get_warp = cute.ffi(name="nvshmemx_qp_int16_get_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get_block = cute.ffi(name="nvshmemx_qp_int16_get_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get = cute.ffi(name="nvshmemx_qp_int32_get", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get_warp = cute.ffi(name="nvshmemx_qp_int32_get_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get_block = cute.ffi(name="nvshmemx_qp_int32_get_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get = cute.ffi(name="nvshmemx_qp_int64_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get_warp = cute.ffi(name="nvshmemx_qp_int64_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get_block = cute.ffi(name="nvshmemx_qp_int64_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get = cute.ffi(name="nvshmemx_qp_uint8_get", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get_warp = cute.ffi(name="nvshmemx_qp_uint8_get_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get_block = cute.ffi(name="nvshmemx_qp_uint8_get_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get = cute.ffi(name="nvshmemx_qp_uint16_get", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get_warp = cute.ffi(name="nvshmemx_qp_uint16_get_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get_block = cute.ffi(name="nvshmemx_qp_uint16_get_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get = cute.ffi(name="nvshmemx_qp_uint32_get", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get_warp = cute.ffi(name="nvshmemx_qp_uint32_get_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get_block = cute.ffi(name="nvshmemx_qp_uint32_get_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get = cute.ffi(name="nvshmemx_qp_uint64_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get_warp = cute.ffi(name="nvshmemx_qp_uint64_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get_block = cute.ffi(name="nvshmemx_qp_uint64_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get = cute.ffi(name="nvshmemx_qp_size_get", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get_warp = cute.ffi(name="nvshmemx_qp_size_get_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get_block = cute.ffi(name="nvshmemx_qp_size_get_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get = cute.ffi(name="nvshmemx_qp_ptrdiff_get", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_get_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get_block = cute.ffi(name="nvshmemx_qp_ptrdiff_get_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_get_nbi = cute.ffi(name="nvshmemx_qp_bfloat16_get_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_get_nbi_warp = cute.ffi(name="nvshmemx_qp_bfloat16_get_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_get_nbi_block = cute.ffi(name="nvshmemx_qp_bfloat16_get_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get_nbi = cute.ffi(name="nvshmemx_qp_half_get_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get_nbi_warp = cute.ffi(name="nvshmemx_qp_half_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_get_nbi_block = cute.ffi(name="nvshmemx_qp_half_get_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get_nbi = cute.ffi(name="nvshmemx_qp_float_get_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get_nbi_warp = cute.ffi(name="nvshmemx_qp_float_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_get_nbi_block = cute.ffi(name="nvshmemx_qp_float_get_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get_nbi = cute.ffi(name="nvshmemx_qp_double_get_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get_nbi_warp = cute.ffi(name="nvshmemx_qp_double_get_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_get_nbi_block = cute.ffi(name="nvshmemx_qp_double_get_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get_nbi = cute.ffi(name="nvshmemx_qp_char_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get_nbi_warp = cute.ffi(name="nvshmemx_qp_char_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_get_nbi_block = cute.ffi(name="nvshmemx_qp_char_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get_nbi = cute.ffi(name="nvshmemx_qp_short_get_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get_nbi_warp = cute.ffi(name="nvshmemx_qp_short_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_get_nbi_block = cute.ffi(name="nvshmemx_qp_short_get_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get_nbi = cute.ffi(name="nvshmemx_qp_schar_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get_nbi_warp = cute.ffi(name="nvshmemx_qp_schar_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_get_nbi_block = cute.ffi(name="nvshmemx_qp_schar_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get_nbi = cute.ffi(name="nvshmemx_qp_int_get_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get_nbi_warp = cute.ffi(name="nvshmemx_qp_int_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_get_nbi_block = cute.ffi(name="nvshmemx_qp_int_get_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get_nbi = cute.ffi(name="nvshmemx_qp_long_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get_nbi_warp = cute.ffi(name="nvshmemx_qp_long_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_get_nbi_block = cute.ffi(name="nvshmemx_qp_long_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get_nbi = cute.ffi(name="nvshmemx_qp_longlong_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get_nbi_warp = cute.ffi(name="nvshmemx_qp_longlong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_get_nbi_block = cute.ffi(name="nvshmemx_qp_longlong_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get_nbi = cute.ffi(name="nvshmemx_qp_uchar_get_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uchar_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_get_nbi_block = cute.ffi(name="nvshmemx_qp_uchar_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get_nbi = cute.ffi(name="nvshmemx_qp_ushort_get_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get_nbi_warp = cute.ffi(name="nvshmemx_qp_ushort_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_get_nbi_block = cute.ffi(name="nvshmemx_qp_ushort_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get_nbi = cute.ffi(name="nvshmemx_qp_uint_get_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uint_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_get_nbi_block = cute.ffi(name="nvshmemx_qp_uint_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get_nbi = cute.ffi(name="nvshmemx_qp_ulong_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get_nbi_warp = cute.ffi(name="nvshmemx_qp_ulong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_get_nbi_block = cute.ffi(name="nvshmemx_qp_ulong_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get_nbi = cute.ffi(name="nvshmemx_qp_ulonglong_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get_nbi_warp = cute.ffi(name="nvshmemx_qp_ulonglong_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_get_nbi_block = cute.ffi(name="nvshmemx_qp_ulonglong_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get_nbi = cute.ffi(name="nvshmemx_qp_int8_get_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get_nbi_warp = cute.ffi(name="nvshmemx_qp_int8_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_get_nbi_block = cute.ffi(name="nvshmemx_qp_int8_get_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get_nbi = cute.ffi(name="nvshmemx_qp_int16_get_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get_nbi_warp = cute.ffi(name="nvshmemx_qp_int16_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_get_nbi_block = cute.ffi(name="nvshmemx_qp_int16_get_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get_nbi = cute.ffi(name="nvshmemx_qp_int32_get_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get_nbi_warp = cute.ffi(name="nvshmemx_qp_int32_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_get_nbi_block = cute.ffi(name="nvshmemx_qp_int32_get_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get_nbi = cute.ffi(name="nvshmemx_qp_int64_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get_nbi_warp = cute.ffi(name="nvshmemx_qp_int64_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_get_nbi_block = cute.ffi(name="nvshmemx_qp_int64_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get_nbi = cute.ffi(name="nvshmemx_qp_uint8_get_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uint8_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_get_nbi_block = cute.ffi(name="nvshmemx_qp_uint8_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get_nbi = cute.ffi(name="nvshmemx_qp_uint16_get_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uint16_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_get_nbi_block = cute.ffi(name="nvshmemx_qp_uint16_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get_nbi = cute.ffi(name="nvshmemx_qp_uint32_get_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uint32_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_get_nbi_block = cute.ffi(name="nvshmemx_qp_uint32_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get_nbi = cute.ffi(name="nvshmemx_qp_uint64_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get_nbi_warp = cute.ffi(name="nvshmemx_qp_uint64_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_get_nbi_block = cute.ffi(name="nvshmemx_qp_uint64_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get_nbi = cute.ffi(name="nvshmemx_qp_size_get_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get_nbi_warp = cute.ffi(name="nvshmemx_qp_size_get_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_get_nbi_block = cute.ffi(name="nvshmemx_qp_size_get_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get_nbi = cute.ffi(name="nvshmemx_qp_ptrdiff_get_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get_nbi_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_get_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_get_nbi_block = cute.ffi(name="nvshmemx_qp_ptrdiff_get_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put = cute.ffi(name="nvshmemx_qp_bfloat16_put", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_warp = cute.ffi(name="nvshmemx_qp_bfloat16_put_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_block = cute.ffi(name="nvshmemx_qp_bfloat16_put_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put = cute.ffi(name="nvshmemx_qp_half_put", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put_warp = cute.ffi(name="nvshmemx_qp_half_put_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put_block = cute.ffi(name="nvshmemx_qp_half_put_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put = cute.ffi(name="nvshmemx_qp_float_put", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put_warp = cute.ffi(name="nvshmemx_qp_float_put_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put_block = cute.ffi(name="nvshmemx_qp_float_put_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put = cute.ffi(name="nvshmemx_qp_double_put", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put_warp = cute.ffi(name="nvshmemx_qp_double_put_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put_block = cute.ffi(name="nvshmemx_qp_double_put_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put = cute.ffi(name="nvshmemx_qp_char_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put_warp = cute.ffi(name="nvshmemx_qp_char_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put_block = cute.ffi(name="nvshmemx_qp_char_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put = cute.ffi(name="nvshmemx_qp_short_put", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put_warp = cute.ffi(name="nvshmemx_qp_short_put_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put_block = cute.ffi(name="nvshmemx_qp_short_put_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put = cute.ffi(name="nvshmemx_qp_schar_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put_warp = cute.ffi(name="nvshmemx_qp_schar_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put_block = cute.ffi(name="nvshmemx_qp_schar_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put = cute.ffi(name="nvshmemx_qp_int_put", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put_warp = cute.ffi(name="nvshmemx_qp_int_put_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put_block = cute.ffi(name="nvshmemx_qp_int_put_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put = cute.ffi(name="nvshmemx_qp_long_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put_warp = cute.ffi(name="nvshmemx_qp_long_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put_block = cute.ffi(name="nvshmemx_qp_long_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put = cute.ffi(name="nvshmemx_qp_longlong_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put_warp = cute.ffi(name="nvshmemx_qp_longlong_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put_block = cute.ffi(name="nvshmemx_qp_longlong_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put = cute.ffi(name="nvshmemx_qp_uchar_put", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put_warp = cute.ffi(name="nvshmemx_qp_uchar_put_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put_block = cute.ffi(name="nvshmemx_qp_uchar_put_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put = cute.ffi(name="nvshmemx_qp_ushort_put", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put_warp = cute.ffi(name="nvshmemx_qp_ushort_put_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put_block = cute.ffi(name="nvshmemx_qp_ushort_put_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put = cute.ffi(name="nvshmemx_qp_uint_put", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put_warp = cute.ffi(name="nvshmemx_qp_uint_put_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put_block = cute.ffi(name="nvshmemx_qp_uint_put_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put = cute.ffi(name="nvshmemx_qp_ulong_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put_warp = cute.ffi(name="nvshmemx_qp_ulong_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put_block = cute.ffi(name="nvshmemx_qp_ulong_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put = cute.ffi(name="nvshmemx_qp_ulonglong_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_warp = cute.ffi(name="nvshmemx_qp_ulonglong_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_block = cute.ffi(name="nvshmemx_qp_ulonglong_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put = cute.ffi(name="nvshmemx_qp_int8_put", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put_warp = cute.ffi(name="nvshmemx_qp_int8_put_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put_block = cute.ffi(name="nvshmemx_qp_int8_put_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put = cute.ffi(name="nvshmemx_qp_int16_put", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put_warp = cute.ffi(name="nvshmemx_qp_int16_put_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put_block = cute.ffi(name="nvshmemx_qp_int16_put_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put = cute.ffi(name="nvshmemx_qp_int32_put", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put_warp = cute.ffi(name="nvshmemx_qp_int32_put_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put_block = cute.ffi(name="nvshmemx_qp_int32_put_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put = cute.ffi(name="nvshmemx_qp_int64_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put_warp = cute.ffi(name="nvshmemx_qp_int64_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put_block = cute.ffi(name="nvshmemx_qp_int64_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put = cute.ffi(name="nvshmemx_qp_uint8_put", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put_warp = cute.ffi(name="nvshmemx_qp_uint8_put_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put_block = cute.ffi(name="nvshmemx_qp_uint8_put_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put = cute.ffi(name="nvshmemx_qp_uint16_put", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put_warp = cute.ffi(name="nvshmemx_qp_uint16_put_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put_block = cute.ffi(name="nvshmemx_qp_uint16_put_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put = cute.ffi(name="nvshmemx_qp_uint32_put", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put_warp = cute.ffi(name="nvshmemx_qp_uint32_put_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put_block = cute.ffi(name="nvshmemx_qp_uint32_put_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put = cute.ffi(name="nvshmemx_qp_uint64_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put_warp = cute.ffi(name="nvshmemx_qp_uint64_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put_block = cute.ffi(name="nvshmemx_qp_uint64_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put = cute.ffi(name="nvshmemx_qp_size_put", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put_warp = cute.ffi(name="nvshmemx_qp_size_put_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put_block = cute.ffi(name="nvshmemx_qp_size_put_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put = cute.ffi(name="nvshmemx_qp_ptrdiff_put", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_put_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_block = cute.ffi(name="nvshmemx_qp_ptrdiff_put_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_nbi = cute.ffi(name="nvshmemx_qp_bfloat16_put_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_nbi_warp = cute.ffi(name="nvshmemx_qp_bfloat16_put_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_nbi_block = cute.ffi(name="nvshmemx_qp_bfloat16_put_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put_nbi = cute.ffi(name="nvshmemx_qp_half_put_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put_nbi_warp = cute.ffi(name="nvshmemx_qp_half_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_half_put_nbi_block = cute.ffi(name="nvshmemx_qp_half_put_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put_nbi = cute.ffi(name="nvshmemx_qp_float_put_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put_nbi_warp = cute.ffi(name="nvshmemx_qp_float_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_float_put_nbi_block = cute.ffi(name="nvshmemx_qp_float_put_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put_nbi = cute.ffi(name="nvshmemx_qp_double_put_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put_nbi_warp = cute.ffi(name="nvshmemx_qp_double_put_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_double_put_nbi_block = cute.ffi(name="nvshmemx_qp_double_put_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put_nbi = cute.ffi(name="nvshmemx_qp_char_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put_nbi_warp = cute.ffi(name="nvshmemx_qp_char_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_char_put_nbi_block = cute.ffi(name="nvshmemx_qp_char_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put_nbi = cute.ffi(name="nvshmemx_qp_short_put_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put_nbi_warp = cute.ffi(name="nvshmemx_qp_short_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_short_put_nbi_block = cute.ffi(name="nvshmemx_qp_short_put_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put_nbi = cute.ffi(name="nvshmemx_qp_schar_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put_nbi_warp = cute.ffi(name="nvshmemx_qp_schar_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_schar_put_nbi_block = cute.ffi(name="nvshmemx_qp_schar_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put_nbi = cute.ffi(name="nvshmemx_qp_int_put_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put_nbi_warp = cute.ffi(name="nvshmemx_qp_int_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int_put_nbi_block = cute.ffi(name="nvshmemx_qp_int_put_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put_nbi = cute.ffi(name="nvshmemx_qp_long_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put_nbi_warp = cute.ffi(name="nvshmemx_qp_long_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_long_put_nbi_block = cute.ffi(name="nvshmemx_qp_long_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put_nbi = cute.ffi(name="nvshmemx_qp_longlong_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put_nbi_warp = cute.ffi(name="nvshmemx_qp_longlong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_longlong_put_nbi_block = cute.ffi(name="nvshmemx_qp_longlong_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put_nbi = cute.ffi(name="nvshmemx_qp_uchar_put_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uchar_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uchar_put_nbi_block = cute.ffi(name="nvshmemx_qp_uchar_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put_nbi = cute.ffi(name="nvshmemx_qp_ushort_put_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put_nbi_warp = cute.ffi(name="nvshmemx_qp_ushort_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ushort_put_nbi_block = cute.ffi(name="nvshmemx_qp_ushort_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put_nbi = cute.ffi(name="nvshmemx_qp_uint_put_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uint_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint_put_nbi_block = cute.ffi(name="nvshmemx_qp_uint_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put_nbi = cute.ffi(name="nvshmemx_qp_ulong_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put_nbi_warp = cute.ffi(name="nvshmemx_qp_ulong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulong_put_nbi_block = cute.ffi(name="nvshmemx_qp_ulong_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_nbi = cute.ffi(name="nvshmemx_qp_ulonglong_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_nbi_warp = cute.ffi(name="nvshmemx_qp_ulonglong_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_nbi_block = cute.ffi(name="nvshmemx_qp_ulonglong_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put_nbi = cute.ffi(name="nvshmemx_qp_int8_put_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put_nbi_warp = cute.ffi(name="nvshmemx_qp_int8_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_put_nbi_block = cute.ffi(name="nvshmemx_qp_int8_put_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put_nbi = cute.ffi(name="nvshmemx_qp_int16_put_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put_nbi_warp = cute.ffi(name="nvshmemx_qp_int16_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int16_put_nbi_block = cute.ffi(name="nvshmemx_qp_int16_put_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put_nbi = cute.ffi(name="nvshmemx_qp_int32_put_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put_nbi_warp = cute.ffi(name="nvshmemx_qp_int32_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int32_put_nbi_block = cute.ffi(name="nvshmemx_qp_int32_put_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put_nbi = cute.ffi(name="nvshmemx_qp_int64_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put_nbi_warp = cute.ffi(name="nvshmemx_qp_int64_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int64_put_nbi_block = cute.ffi(name="nvshmemx_qp_int64_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put_nbi = cute.ffi(name="nvshmemx_qp_uint8_put_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uint8_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint8_put_nbi_block = cute.ffi(name="nvshmemx_qp_uint8_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put_nbi = cute.ffi(name="nvshmemx_qp_uint16_put_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uint16_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint16_put_nbi_block = cute.ffi(name="nvshmemx_qp_uint16_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put_nbi = cute.ffi(name="nvshmemx_qp_uint32_put_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uint32_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint32_put_nbi_block = cute.ffi(name="nvshmemx_qp_uint32_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put_nbi = cute.ffi(name="nvshmemx_qp_uint64_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put_nbi_warp = cute.ffi(name="nvshmemx_qp_uint64_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_uint64_put_nbi_block = cute.ffi(name="nvshmemx_qp_uint64_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put_nbi = cute.ffi(name="nvshmemx_qp_size_put_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put_nbi_warp = cute.ffi(name="nvshmemx_qp_size_put_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_put_nbi_block = cute.ffi(name="nvshmemx_qp_size_put_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_nbi = cute.ffi(name="nvshmemx_qp_ptrdiff_put_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_nbi_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_put_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_nbi_block = cute.ffi(name="nvshmemx_qp_ptrdiff_put_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_p = cute.ffi(name="nvshmemx_qp_bfloat16_p", params_types=[_CutePtrType(cutlass.BFloat16), cutlass.BFloat16, cutlass.Int32, cutlass.Int32])

qp_half_p = cute.ffi(name="nvshmemx_qp_half_p", params_types=[_CutePtrType(cutlass.Float16), cutlass.Float16, cutlass.Int32, cutlass.Int32])

qp_float_p = cute.ffi(name="nvshmemx_qp_float_p", params_types=[_CutePtrType(cutlass.Float32), cutlass.Float32, cutlass.Int32, cutlass.Int32])

qp_double_p = cute.ffi(name="nvshmemx_qp_double_p", params_types=[_CutePtrType(cutlass.Float64), cutlass.Float64, cutlass.Int32, cutlass.Int32])

qp_char_p = cute.ffi(name="nvshmemx_qp_char_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32, cutlass.Int32])

qp_short_p = cute.ffi(name="nvshmemx_qp_short_p", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int16, cutlass.Int32, cutlass.Int32])

qp_schar_p = cute.ffi(name="nvshmemx_qp_schar_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32, cutlass.Int32])

qp_int_p = cute.ffi(name="nvshmemx_qp_int_p", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_p = cute.ffi(name="nvshmemx_qp_long_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32, cutlass.Int32])

qp_longlong_p = cute.ffi(name="nvshmemx_qp_longlong_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32, cutlass.Int32])

qp_uchar_p = cute.ffi(name="nvshmemx_qp_uchar_p", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Uint8, cutlass.Int32, cutlass.Int32])

qp_ushort_p = cute.ffi(name="nvshmemx_qp_ushort_p", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint16, cutlass.Int32, cutlass.Int32])

qp_uint_p = cute.ffi(name="nvshmemx_qp_uint_p", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32, cutlass.Int32])

qp_ulong_p = cute.ffi(name="nvshmemx_qp_ulong_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ulonglong_p = cute.ffi(name="nvshmemx_qp_ulonglong_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_int8_p = cute.ffi(name="nvshmemx_qp_int8_p", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int8, cutlass.Int32, cutlass.Int32])

qp_int16_p = cute.ffi(name="nvshmemx_qp_int16_p", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int16, cutlass.Int32, cutlass.Int32])

qp_int32_p = cute.ffi(name="nvshmemx_qp_int32_p", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_p = cute.ffi(name="nvshmemx_qp_int64_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32, cutlass.Int32])

qp_uint8_p = cute.ffi(name="nvshmemx_qp_uint8_p", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Uint8, cutlass.Int32, cutlass.Int32])

qp_uint16_p = cute.ffi(name="nvshmemx_qp_uint16_p", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Uint16, cutlass.Int32, cutlass.Int32])

qp_uint32_p = cute.ffi(name="nvshmemx_qp_uint32_p", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Uint32, cutlass.Int32, cutlass.Int32])

qp_uint64_p = cute.ffi(name="nvshmemx_qp_uint64_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_size_p = cute.ffi(name="nvshmemx_qp_size_p", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_p = cute.ffi(name="nvshmemx_qp_ptrdiff_p", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int64, cutlass.Int32, cutlass.Int32])

qp_bfloat16_g = cute.ffi(name="nvshmemx_qp_bfloat16_g", params_types=[_CutePtrType(cutlass.BFloat16), cutlass.Int32, cutlass.Int32], return_type=cutlass.BFloat16)

qp_half_g = cute.ffi(name="nvshmemx_qp_half_g", params_types=[_CutePtrType(cutlass.Float16), cutlass.Int32, cutlass.Int32], return_type=cutlass.Float16)

qp_float_g = cute.ffi(name="nvshmemx_qp_float_g", params_types=[_CutePtrType(cutlass.Float32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Float32)

qp_double_g = cute.ffi(name="nvshmemx_qp_double_g", params_types=[_CutePtrType(cutlass.Float64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Float64)

qp_char_g = cute.ffi(name="nvshmemx_qp_char_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int8)

qp_short_g = cute.ffi(name="nvshmemx_qp_short_g", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int16)

qp_schar_g = cute.ffi(name="nvshmemx_qp_schar_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int8)

qp_int_g = cute.ffi(name="nvshmemx_qp_int_g", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

qp_long_g = cute.ffi(name="nvshmemx_qp_long_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int64)

qp_longlong_g = cute.ffi(name="nvshmemx_qp_longlong_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int64)

qp_uchar_g = cute.ffi(name="nvshmemx_qp_uchar_g", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint8)

qp_ushort_g = cute.ffi(name="nvshmemx_qp_ushort_g", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint16)

qp_uint_g = cute.ffi(name="nvshmemx_qp_uint_g", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint32)

qp_ulong_g = cute.ffi(name="nvshmemx_qp_ulong_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

qp_ulonglong_g = cute.ffi(name="nvshmemx_qp_ulonglong_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

qp_int8_g = cute.ffi(name="nvshmemx_qp_int8_g", params_types=[_CutePtrType(cutlass.Int8), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int8)

qp_int16_g = cute.ffi(name="nvshmemx_qp_int16_g", params_types=[_CutePtrType(cutlass.Int16), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int16)

qp_int32_g = cute.ffi(name="nvshmemx_qp_int32_g", params_types=[_CutePtrType(cutlass.Int32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int32)

qp_int64_g = cute.ffi(name="nvshmemx_qp_int64_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int64)

qp_uint8_g = cute.ffi(name="nvshmemx_qp_uint8_g", params_types=[_CutePtrType(cutlass.Uint8), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint8)

qp_uint16_g = cute.ffi(name="nvshmemx_qp_uint16_g", params_types=[_CutePtrType(cutlass.Uint16), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint16)

qp_uint32_g = cute.ffi(name="nvshmemx_qp_uint32_g", params_types=[_CutePtrType(cutlass.Uint32), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint32)

qp_uint64_g = cute.ffi(name="nvshmemx_qp_uint64_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

qp_size_g = cute.ffi(name="nvshmemx_qp_size_g", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Uint64)

qp_ptrdiff_g = cute.ffi(name="nvshmemx_qp_ptrdiff_g", params_types=[_CutePtrType(cutlass.Int64), cutlass.Int32, cutlass.Int32], return_type=cutlass.Int64)

qp_signal_op = cute.ffi(name="nvshmemx_qp_signal_op", params_types=[_CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal_warp = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal_block = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal = cute.ffi(name="nvshmemx_qp_half_put_signal", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal_warp = cute.ffi(name="nvshmemx_qp_half_put_signal_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal_block = cute.ffi(name="nvshmemx_qp_half_put_signal_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal = cute.ffi(name="nvshmemx_qp_float_put_signal", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal_warp = cute.ffi(name="nvshmemx_qp_float_put_signal_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal_block = cute.ffi(name="nvshmemx_qp_float_put_signal_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal = cute.ffi(name="nvshmemx_qp_double_put_signal", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal_warp = cute.ffi(name="nvshmemx_qp_double_put_signal_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal_block = cute.ffi(name="nvshmemx_qp_double_put_signal_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal = cute.ffi(name="nvshmemx_qp_char_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal_warp = cute.ffi(name="nvshmemx_qp_char_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal_block = cute.ffi(name="nvshmemx_qp_char_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal = cute.ffi(name="nvshmemx_qp_short_put_signal", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal_warp = cute.ffi(name="nvshmemx_qp_short_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal_block = cute.ffi(name="nvshmemx_qp_short_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal = cute.ffi(name="nvshmemx_qp_schar_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal_warp = cute.ffi(name="nvshmemx_qp_schar_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal_block = cute.ffi(name="nvshmemx_qp_schar_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal = cute.ffi(name="nvshmemx_qp_int_put_signal", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal_warp = cute.ffi(name="nvshmemx_qp_int_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal_block = cute.ffi(name="nvshmemx_qp_int_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal = cute.ffi(name="nvshmemx_qp_long_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal_warp = cute.ffi(name="nvshmemx_qp_long_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal_block = cute.ffi(name="nvshmemx_qp_long_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal = cute.ffi(name="nvshmemx_qp_longlong_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal_warp = cute.ffi(name="nvshmemx_qp_longlong_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal_block = cute.ffi(name="nvshmemx_qp_longlong_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal = cute.ffi(name="nvshmemx_qp_uchar_put_signal", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal_warp = cute.ffi(name="nvshmemx_qp_uchar_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal_block = cute.ffi(name="nvshmemx_qp_uchar_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal = cute.ffi(name="nvshmemx_qp_ushort_put_signal", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal_warp = cute.ffi(name="nvshmemx_qp_ushort_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal_block = cute.ffi(name="nvshmemx_qp_ushort_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal = cute.ffi(name="nvshmemx_qp_uint_put_signal", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal_warp = cute.ffi(name="nvshmemx_qp_uint_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal_block = cute.ffi(name="nvshmemx_qp_uint_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal = cute.ffi(name="nvshmemx_qp_ulong_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal_warp = cute.ffi(name="nvshmemx_qp_ulong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal_block = cute.ffi(name="nvshmemx_qp_ulong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal_warp = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal_block = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal = cute.ffi(name="nvshmemx_qp_int8_put_signal", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal_warp = cute.ffi(name="nvshmemx_qp_int8_put_signal_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal_block = cute.ffi(name="nvshmemx_qp_int8_put_signal_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal = cute.ffi(name="nvshmemx_qp_int16_put_signal", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal_warp = cute.ffi(name="nvshmemx_qp_int16_put_signal_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal_block = cute.ffi(name="nvshmemx_qp_int16_put_signal_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal = cute.ffi(name="nvshmemx_qp_int32_put_signal", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal_warp = cute.ffi(name="nvshmemx_qp_int32_put_signal_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal_block = cute.ffi(name="nvshmemx_qp_int32_put_signal_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal = cute.ffi(name="nvshmemx_qp_int64_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal_warp = cute.ffi(name="nvshmemx_qp_int64_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal_block = cute.ffi(name="nvshmemx_qp_int64_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal = cute.ffi(name="nvshmemx_qp_uint8_put_signal", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal_warp = cute.ffi(name="nvshmemx_qp_uint8_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal_block = cute.ffi(name="nvshmemx_qp_uint8_put_signal_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal = cute.ffi(name="nvshmemx_qp_uint16_put_signal", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal_warp = cute.ffi(name="nvshmemx_qp_uint16_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal_block = cute.ffi(name="nvshmemx_qp_uint16_put_signal_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal = cute.ffi(name="nvshmemx_qp_uint32_put_signal", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal_warp = cute.ffi(name="nvshmemx_qp_uint32_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal_block = cute.ffi(name="nvshmemx_qp_uint32_put_signal_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal = cute.ffi(name="nvshmemx_qp_uint64_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal_warp = cute.ffi(name="nvshmemx_qp_uint64_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal_block = cute.ffi(name="nvshmemx_qp_uint64_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal = cute.ffi(name="nvshmemx_qp_size_put_signal", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal_warp = cute.ffi(name="nvshmemx_qp_size_put_signal_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal_block = cute.ffi(name="nvshmemx_qp_size_put_signal_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal_block = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal_nbi = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal_nbi", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_bfloat16_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_bfloat16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal_nbi = cute.ffi(name="nvshmemx_qp_half_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_half_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_half_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_half_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal_nbi = cute.ffi(name="nvshmemx_qp_float_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_float_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_float_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_float_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal_nbi = cute.ffi(name="nvshmemx_qp_double_put_signal_nbi", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_double_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_double_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_double_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal_nbi = cute.ffi(name="nvshmemx_qp_char_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_char_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_char_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_char_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal_nbi = cute.ffi(name="nvshmemx_qp_short_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_short_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_short_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_short_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal_nbi = cute.ffi(name="nvshmemx_qp_schar_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_schar_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_schar_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_schar_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal_nbi = cute.ffi(name="nvshmemx_qp_int_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_int_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_int_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal_nbi = cute.ffi(name="nvshmemx_qp_long_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_long_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_long_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_long_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal_nbi = cute.ffi(name="nvshmemx_qp_longlong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_longlong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_longlong_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_longlong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uchar_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uchar_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uchar_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uchar_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal_nbi = cute.ffi(name="nvshmemx_qp_ushort_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_ushort_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ushort_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_ushort_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uint_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uint_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uint_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal_nbi = cute.ffi(name="nvshmemx_qp_ulong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_ulong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulong_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_ulong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal_nbi = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ulonglong_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_ulonglong_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal_nbi = cute.ffi(name="nvshmemx_qp_int8_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_int8_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int8_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_int8_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal_nbi = cute.ffi(name="nvshmemx_qp_int16_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_int16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int16_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_int16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal_nbi = cute.ffi(name="nvshmemx_qp_int32_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_int32_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int32_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_int32_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal_nbi = cute.ffi(name="nvshmemx_qp_int64_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_int64_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_int64_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_int64_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uint8_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uint8_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint8_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uint8_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uint16_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uint16_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint16_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uint16_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uint32_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uint32_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint32_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uint32_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal_nbi = cute.ffi(name="nvshmemx_qp_uint64_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_uint64_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_uint64_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_uint64_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal_nbi = cute.ffi(name="nvshmemx_qp_size_put_signal_nbi", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_size_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_size_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_size_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal_nbi = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal_nbi", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal_nbi_warp = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal_nbi_warp", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_ptrdiff_put_signal_nbi_block = cute.ffi(name="nvshmemx_qp_ptrdiff_put_signal_nbi_block", params_types=[_CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Int32])

qp_quiet = cute.ffi(name="nvshmemx_qp_quiet", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

qp_quiet_warp = cute.ffi(name="nvshmemx_qp_quiet_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

qp_quiet_block = cute.ffi(name="nvshmemx_qp_quiet_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

qp_fence = cute.ffi(name="nvshmemx_qp_fence", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

qp_fence_warp = cute.ffi(name="nvshmemx_qp_fence_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

qp_fence_block = cute.ffi(name="nvshmemx_qp_fence_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), cutlass.Int32])

alltoallmem_warp = cute.ffi(name="nvshmemx_alltoallmem_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

alltoallmem_block = cute.ffi(name="nvshmemx_alltoallmem_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_alltoall_warp = cute.ffi(name="nvshmemx_bfloat16_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_alltoall_warp = cute.ffi(name="nvshmemx_half_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_alltoall_warp = cute.ffi(name="nvshmemx_float_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_alltoall_warp = cute.ffi(name="nvshmemx_double_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_alltoall_warp = cute.ffi(name="nvshmemx_char_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_alltoall_warp = cute.ffi(name="nvshmemx_short_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_alltoall_warp = cute.ffi(name="nvshmemx_schar_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_alltoall_warp = cute.ffi(name="nvshmemx_int_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_alltoall_warp = cute.ffi(name="nvshmemx_long_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_alltoall_warp = cute.ffi(name="nvshmemx_longlong_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_alltoall_warp = cute.ffi(name="nvshmemx_uchar_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_alltoall_warp = cute.ffi(name="nvshmemx_ushort_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_alltoall_warp = cute.ffi(name="nvshmemx_uint_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_alltoall_warp = cute.ffi(name="nvshmemx_ulong_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_alltoall_warp = cute.ffi(name="nvshmemx_ulonglong_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_alltoall_warp = cute.ffi(name="nvshmemx_int8_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_alltoall_warp = cute.ffi(name="nvshmemx_int16_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_alltoall_warp = cute.ffi(name="nvshmemx_int32_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_alltoall_warp = cute.ffi(name="nvshmemx_int64_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_alltoall_warp = cute.ffi(name="nvshmemx_uint8_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_alltoall_warp = cute.ffi(name="nvshmemx_uint16_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_alltoall_warp = cute.ffi(name="nvshmemx_uint32_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_alltoall_warp = cute.ffi(name="nvshmemx_uint64_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_alltoall_warp = cute.ffi(name="nvshmemx_size_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_alltoall_warp = cute.ffi(name="nvshmemx_ptrdiff_alltoall_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_alltoall_block = cute.ffi(name="nvshmemx_bfloat16_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_alltoall_block = cute.ffi(name="nvshmemx_half_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_alltoall_block = cute.ffi(name="nvshmemx_float_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_alltoall_block = cute.ffi(name="nvshmemx_double_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_alltoall_block = cute.ffi(name="nvshmemx_char_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_alltoall_block = cute.ffi(name="nvshmemx_short_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_alltoall_block = cute.ffi(name="nvshmemx_schar_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_alltoall_block = cute.ffi(name="nvshmemx_int_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_alltoall_block = cute.ffi(name="nvshmemx_long_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_alltoall_block = cute.ffi(name="nvshmemx_longlong_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_alltoall_block = cute.ffi(name="nvshmemx_uchar_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_alltoall_block = cute.ffi(name="nvshmemx_ushort_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_alltoall_block = cute.ffi(name="nvshmemx_uint_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_alltoall_block = cute.ffi(name="nvshmemx_ulong_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_alltoall_block = cute.ffi(name="nvshmemx_ulonglong_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_alltoall_block = cute.ffi(name="nvshmemx_int8_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_alltoall_block = cute.ffi(name="nvshmemx_int16_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_alltoall_block = cute.ffi(name="nvshmemx_int32_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_alltoall_block = cute.ffi(name="nvshmemx_int64_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_alltoall_block = cute.ffi(name="nvshmemx_uint8_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_alltoall_block = cute.ffi(name="nvshmemx_uint16_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_alltoall_block = cute.ffi(name="nvshmemx_uint32_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_alltoall_block = cute.ffi(name="nvshmemx_uint64_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_alltoall_block = cute.ffi(name="nvshmemx_size_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_alltoall_block = cute.ffi(name="nvshmemx_ptrdiff_alltoall_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

barrier_warp = cute.ffi(name="nvshmemx_barrier_warp", params_types=[cutlass.Int32], return_type=cutlass.Int32)

barrier_warpgroup = cute.ffi(name="nvshmemx_barrier_warpgroup", params_types=[cutlass.Int32], return_type=cutlass.Int32)

barrier_block = cute.ffi(name="nvshmemx_barrier_block", params_types=[cutlass.Int32], return_type=cutlass.Int32)

barrier_all_warp = cute.ffi(name="nvshmemx_barrier_all_warp", params_types=[])

barrier_all_block = cute.ffi(name="nvshmemx_barrier_all_block", params_types=[])

team_sync_warp = cute.ffi(name="nvshmemx_team_sync_warp", params_types=[cutlass.Int32], return_type=cutlass.Int32)

team_sync_block = cute.ffi(name="nvshmemx_team_sync_block", params_types=[cutlass.Int32], return_type=cutlass.Int32)

sync_all_warp = cute.ffi(name="nvshmemx_sync_all_warp", params_types=[])

sync_all_block = cute.ffi(name="nvshmemx_sync_all_block", params_types=[])

broadcastmem_warp = cute.ffi(name="nvshmemx_broadcastmem_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

broadcastmem_block = cute.ffi(name="nvshmemx_broadcastmem_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

bfloat16_broadcast_warp = cute.ffi(name="nvshmemx_bfloat16_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

half_broadcast_warp = cute.ffi(name="nvshmemx_half_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

float_broadcast_warp = cute.ffi(name="nvshmemx_float_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

double_broadcast_warp = cute.ffi(name="nvshmemx_double_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

char_broadcast_warp = cute.ffi(name="nvshmemx_char_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

short_broadcast_warp = cute.ffi(name="nvshmemx_short_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

schar_broadcast_warp = cute.ffi(name="nvshmemx_schar_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int_broadcast_warp = cute.ffi(name="nvshmemx_int_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

long_broadcast_warp = cute.ffi(name="nvshmemx_long_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

longlong_broadcast_warp = cute.ffi(name="nvshmemx_longlong_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uchar_broadcast_warp = cute.ffi(name="nvshmemx_uchar_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ushort_broadcast_warp = cute.ffi(name="nvshmemx_ushort_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint_broadcast_warp = cute.ffi(name="nvshmemx_uint_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulong_broadcast_warp = cute.ffi(name="nvshmemx_ulong_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulonglong_broadcast_warp = cute.ffi(name="nvshmemx_ulonglong_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int8_broadcast_warp = cute.ffi(name="nvshmemx_int8_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int16_broadcast_warp = cute.ffi(name="nvshmemx_int16_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int32_broadcast_warp = cute.ffi(name="nvshmemx_int32_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int64_broadcast_warp = cute.ffi(name="nvshmemx_int64_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint8_broadcast_warp = cute.ffi(name="nvshmemx_uint8_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint16_broadcast_warp = cute.ffi(name="nvshmemx_uint16_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint32_broadcast_warp = cute.ffi(name="nvshmemx_uint32_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint64_broadcast_warp = cute.ffi(name="nvshmemx_uint64_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

size_broadcast_warp = cute.ffi(name="nvshmemx_size_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ptrdiff_broadcast_warp = cute.ffi(name="nvshmemx_ptrdiff_broadcast_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

bfloat16_broadcast_block = cute.ffi(name="nvshmemx_bfloat16_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

half_broadcast_block = cute.ffi(name="nvshmemx_half_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

float_broadcast_block = cute.ffi(name="nvshmemx_float_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

double_broadcast_block = cute.ffi(name="nvshmemx_double_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

char_broadcast_block = cute.ffi(name="nvshmemx_char_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

short_broadcast_block = cute.ffi(name="nvshmemx_short_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

schar_broadcast_block = cute.ffi(name="nvshmemx_schar_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int_broadcast_block = cute.ffi(name="nvshmemx_int_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

long_broadcast_block = cute.ffi(name="nvshmemx_long_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

longlong_broadcast_block = cute.ffi(name="nvshmemx_longlong_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uchar_broadcast_block = cute.ffi(name="nvshmemx_uchar_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ushort_broadcast_block = cute.ffi(name="nvshmemx_ushort_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint_broadcast_block = cute.ffi(name="nvshmemx_uint_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulong_broadcast_block = cute.ffi(name="nvshmemx_ulong_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ulonglong_broadcast_block = cute.ffi(name="nvshmemx_ulonglong_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int8_broadcast_block = cute.ffi(name="nvshmemx_int8_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int16_broadcast_block = cute.ffi(name="nvshmemx_int16_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int32_broadcast_block = cute.ffi(name="nvshmemx_int32_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

int64_broadcast_block = cute.ffi(name="nvshmemx_int64_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint8_broadcast_block = cute.ffi(name="nvshmemx_uint8_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint16_broadcast_block = cute.ffi(name="nvshmemx_uint16_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint32_broadcast_block = cute.ffi(name="nvshmemx_uint32_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

uint64_broadcast_block = cute.ffi(name="nvshmemx_uint64_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

size_broadcast_block = cute.ffi(name="nvshmemx_size_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

ptrdiff_broadcast_block = cute.ffi(name="nvshmemx_ptrdiff_broadcast_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64, cutlass.Int32], return_type=cutlass.Int32)

fcollectmem_warp = cute.ffi(name="nvshmemx_fcollectmem_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

fcollectmem_block = cute.ffi(name="nvshmemx_fcollectmem_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_fcollect_warp = cute.ffi(name="nvshmemx_bfloat16_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_fcollect_warp = cute.ffi(name="nvshmemx_half_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_fcollect_warp = cute.ffi(name="nvshmemx_float_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_fcollect_warp = cute.ffi(name="nvshmemx_double_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_fcollect_warp = cute.ffi(name="nvshmemx_char_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_fcollect_warp = cute.ffi(name="nvshmemx_short_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_fcollect_warp = cute.ffi(name="nvshmemx_schar_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_fcollect_warp = cute.ffi(name="nvshmemx_int_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_fcollect_warp = cute.ffi(name="nvshmemx_long_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_fcollect_warp = cute.ffi(name="nvshmemx_longlong_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_fcollect_warp = cute.ffi(name="nvshmemx_uchar_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_fcollect_warp = cute.ffi(name="nvshmemx_ushort_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_fcollect_warp = cute.ffi(name="nvshmemx_uint_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_fcollect_warp = cute.ffi(name="nvshmemx_ulong_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_fcollect_warp = cute.ffi(name="nvshmemx_ulonglong_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_fcollect_warp = cute.ffi(name="nvshmemx_int8_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_fcollect_warp = cute.ffi(name="nvshmemx_int16_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_fcollect_warp = cute.ffi(name="nvshmemx_int32_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_fcollect_warp = cute.ffi(name="nvshmemx_int64_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_fcollect_warp = cute.ffi(name="nvshmemx_uint8_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_fcollect_warp = cute.ffi(name="nvshmemx_uint16_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_fcollect_warp = cute.ffi(name="nvshmemx_uint32_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_fcollect_warp = cute.ffi(name="nvshmemx_uint64_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_fcollect_warp = cute.ffi(name="nvshmemx_size_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_fcollect_warp = cute.ffi(name="nvshmemx_ptrdiff_fcollect_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_fcollect_block = cute.ffi(name="nvshmemx_bfloat16_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_fcollect_block = cute.ffi(name="nvshmemx_half_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_fcollect_block = cute.ffi(name="nvshmemx_float_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_fcollect_block = cute.ffi(name="nvshmemx_double_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

char_fcollect_block = cute.ffi(name="nvshmemx_char_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_fcollect_block = cute.ffi(name="nvshmemx_short_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

schar_fcollect_block = cute.ffi(name="nvshmemx_schar_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int_fcollect_block = cute.ffi(name="nvshmemx_int_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_fcollect_block = cute.ffi(name="nvshmemx_long_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_fcollect_block = cute.ffi(name="nvshmemx_longlong_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_fcollect_block = cute.ffi(name="nvshmemx_uchar_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_fcollect_block = cute.ffi(name="nvshmemx_ushort_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_fcollect_block = cute.ffi(name="nvshmemx_uint_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_fcollect_block = cute.ffi(name="nvshmemx_ulong_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_fcollect_block = cute.ffi(name="nvshmemx_ulonglong_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_fcollect_block = cute.ffi(name="nvshmemx_int8_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_fcollect_block = cute.ffi(name="nvshmemx_int16_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_fcollect_block = cute.ffi(name="nvshmemx_int32_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_fcollect_block = cute.ffi(name="nvshmemx_int64_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_fcollect_block = cute.ffi(name="nvshmemx_uint8_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_fcollect_block = cute.ffi(name="nvshmemx_uint16_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_fcollect_block = cute.ffi(name="nvshmemx_uint32_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_fcollect_block = cute.ffi(name="nvshmemx_uint64_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_fcollect_block = cute.ffi(name="nvshmemx_size_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ptrdiff_fcollect_block = cute.ffi(name="nvshmemx_ptrdiff_fcollect_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reduce_warp = cute.ffi(name="nvshmemx_uchar_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reduce_warp = cute.ffi(name="nvshmemx_ushort_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reduce_warp = cute.ffi(name="nvshmemx_uint_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reduce_warp = cute.ffi(name="nvshmemx_ulong_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reduce_warp = cute.ffi(name="nvshmemx_int8_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reduce_warp = cute.ffi(name="nvshmemx_int16_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reduce_warp = cute.ffi(name="nvshmemx_int32_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reduce_warp = cute.ffi(name="nvshmemx_int64_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reduce_warp = cute.ffi(name="nvshmemx_uint8_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reduce_warp = cute.ffi(name="nvshmemx_uint16_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reduce_warp = cute.ffi(name="nvshmemx_uint32_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reduce_warp = cute.ffi(name="nvshmemx_uint64_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reduce_warp = cute.ffi(name="nvshmemx_size_and_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reduce_warp = cute.ffi(name="nvshmemx_uchar_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reduce_warp = cute.ffi(name="nvshmemx_ushort_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reduce_warp = cute.ffi(name="nvshmemx_uint_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reduce_warp = cute.ffi(name="nvshmemx_ulong_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reduce_warp = cute.ffi(name="nvshmemx_int8_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reduce_warp = cute.ffi(name="nvshmemx_int16_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reduce_warp = cute.ffi(name="nvshmemx_int32_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reduce_warp = cute.ffi(name="nvshmemx_int64_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reduce_warp = cute.ffi(name="nvshmemx_uint8_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reduce_warp = cute.ffi(name="nvshmemx_uint16_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reduce_warp = cute.ffi(name="nvshmemx_uint32_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reduce_warp = cute.ffi(name="nvshmemx_uint64_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reduce_warp = cute.ffi(name="nvshmemx_size_or_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reduce_warp = cute.ffi(name="nvshmemx_uchar_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reduce_warp = cute.ffi(name="nvshmemx_ushort_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reduce_warp = cute.ffi(name="nvshmemx_uint_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reduce_warp = cute.ffi(name="nvshmemx_ulong_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reduce_warp = cute.ffi(name="nvshmemx_int8_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reduce_warp = cute.ffi(name="nvshmemx_int16_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reduce_warp = cute.ffi(name="nvshmemx_int32_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reduce_warp = cute.ffi(name="nvshmemx_int64_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reduce_warp = cute.ffi(name="nvshmemx_uint8_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reduce_warp = cute.ffi(name="nvshmemx_uint16_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reduce_warp = cute.ffi(name="nvshmemx_uint32_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reduce_warp = cute.ffi(name="nvshmemx_uint64_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reduce_warp = cute.ffi(name="nvshmemx_size_xor_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reduce_warp = cute.ffi(name="nvshmemx_uchar_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reduce_warp = cute.ffi(name="nvshmemx_ushort_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reduce_warp = cute.ffi(name="nvshmemx_uint_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reduce_warp = cute.ffi(name="nvshmemx_ulong_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reduce_warp = cute.ffi(name="nvshmemx_int8_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reduce_warp = cute.ffi(name="nvshmemx_int16_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reduce_warp = cute.ffi(name="nvshmemx_int32_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reduce_warp = cute.ffi(name="nvshmemx_int64_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reduce_warp = cute.ffi(name="nvshmemx_uint8_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reduce_warp = cute.ffi(name="nvshmemx_uint16_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reduce_warp = cute.ffi(name="nvshmemx_uint32_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reduce_warp = cute.ffi(name="nvshmemx_uint64_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reduce_warp = cute.ffi(name="nvshmemx_size_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reduce_warp = cute.ffi(name="nvshmemx_char_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reduce_warp = cute.ffi(name="nvshmemx_schar_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reduce_warp = cute.ffi(name="nvshmemx_short_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reduce_warp = cute.ffi(name="nvshmemx_int_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reduce_warp = cute.ffi(name="nvshmemx_long_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reduce_warp = cute.ffi(name="nvshmemx_longlong_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reduce_warp = cute.ffi(name="nvshmemx_bfloat16_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reduce_warp = cute.ffi(name="nvshmemx_half_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reduce_warp = cute.ffi(name="nvshmemx_float_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reduce_warp = cute.ffi(name="nvshmemx_double_max_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reduce_warp = cute.ffi(name="nvshmemx_uchar_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reduce_warp = cute.ffi(name="nvshmemx_ushort_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reduce_warp = cute.ffi(name="nvshmemx_uint_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reduce_warp = cute.ffi(name="nvshmemx_ulong_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reduce_warp = cute.ffi(name="nvshmemx_int8_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reduce_warp = cute.ffi(name="nvshmemx_int16_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reduce_warp = cute.ffi(name="nvshmemx_int32_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reduce_warp = cute.ffi(name="nvshmemx_int64_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reduce_warp = cute.ffi(name="nvshmemx_uint8_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reduce_warp = cute.ffi(name="nvshmemx_uint16_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reduce_warp = cute.ffi(name="nvshmemx_uint32_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reduce_warp = cute.ffi(name="nvshmemx_uint64_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reduce_warp = cute.ffi(name="nvshmemx_size_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reduce_warp = cute.ffi(name="nvshmemx_char_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reduce_warp = cute.ffi(name="nvshmemx_schar_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reduce_warp = cute.ffi(name="nvshmemx_short_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reduce_warp = cute.ffi(name="nvshmemx_int_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reduce_warp = cute.ffi(name="nvshmemx_long_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reduce_warp = cute.ffi(name="nvshmemx_longlong_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reduce_warp = cute.ffi(name="nvshmemx_bfloat16_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reduce_warp = cute.ffi(name="nvshmemx_half_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reduce_warp = cute.ffi(name="nvshmemx_float_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reduce_warp = cute.ffi(name="nvshmemx_double_min_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reduce_warp = cute.ffi(name="nvshmemx_uchar_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reduce_warp = cute.ffi(name="nvshmemx_ushort_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reduce_warp = cute.ffi(name="nvshmemx_uint_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reduce_warp = cute.ffi(name="nvshmemx_ulong_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reduce_warp = cute.ffi(name="nvshmemx_int8_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reduce_warp = cute.ffi(name="nvshmemx_int16_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reduce_warp = cute.ffi(name="nvshmemx_int32_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reduce_warp = cute.ffi(name="nvshmemx_int64_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reduce_warp = cute.ffi(name="nvshmemx_uint8_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reduce_warp = cute.ffi(name="nvshmemx_uint16_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reduce_warp = cute.ffi(name="nvshmemx_uint32_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reduce_warp = cute.ffi(name="nvshmemx_uint64_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reduce_warp = cute.ffi(name="nvshmemx_size_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reduce_warp = cute.ffi(name="nvshmemx_char_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reduce_warp = cute.ffi(name="nvshmemx_schar_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reduce_warp = cute.ffi(name="nvshmemx_short_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reduce_warp = cute.ffi(name="nvshmemx_int_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reduce_warp = cute.ffi(name="nvshmemx_long_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reduce_warp = cute.ffi(name="nvshmemx_longlong_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reduce_warp = cute.ffi(name="nvshmemx_bfloat16_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reduce_warp = cute.ffi(name="nvshmemx_half_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reduce_warp = cute.ffi(name="nvshmemx_float_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reduce_warp = cute.ffi(name="nvshmemx_double_sum_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reduce_warp = cute.ffi(name="nvshmemx_uchar_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reduce_warp = cute.ffi(name="nvshmemx_ushort_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reduce_warp = cute.ffi(name="nvshmemx_uint_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reduce_warp = cute.ffi(name="nvshmemx_ulong_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reduce_warp = cute.ffi(name="nvshmemx_ulonglong_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reduce_warp = cute.ffi(name="nvshmemx_int8_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reduce_warp = cute.ffi(name="nvshmemx_int16_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reduce_warp = cute.ffi(name="nvshmemx_int32_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reduce_warp = cute.ffi(name="nvshmemx_int64_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reduce_warp = cute.ffi(name="nvshmemx_uint8_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reduce_warp = cute.ffi(name="nvshmemx_uint16_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reduce_warp = cute.ffi(name="nvshmemx_uint32_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reduce_warp = cute.ffi(name="nvshmemx_uint64_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reduce_warp = cute.ffi(name="nvshmemx_size_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reduce_warp = cute.ffi(name="nvshmemx_char_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reduce_warp = cute.ffi(name="nvshmemx_schar_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reduce_warp = cute.ffi(name="nvshmemx_short_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reduce_warp = cute.ffi(name="nvshmemx_int_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reduce_warp = cute.ffi(name="nvshmemx_long_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reduce_warp = cute.ffi(name="nvshmemx_longlong_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reduce_warp = cute.ffi(name="nvshmemx_bfloat16_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reduce_warp = cute.ffi(name="nvshmemx_half_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reduce_warp = cute.ffi(name="nvshmemx_float_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reduce_warp = cute.ffi(name="nvshmemx_double_prod_reduce_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reduce_block = cute.ffi(name="nvshmemx_uchar_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reduce_block = cute.ffi(name="nvshmemx_ushort_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reduce_block = cute.ffi(name="nvshmemx_uint_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reduce_block = cute.ffi(name="nvshmemx_ulong_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reduce_block = cute.ffi(name="nvshmemx_ulonglong_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reduce_block = cute.ffi(name="nvshmemx_int8_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reduce_block = cute.ffi(name="nvshmemx_int16_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reduce_block = cute.ffi(name="nvshmemx_int32_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reduce_block = cute.ffi(name="nvshmemx_int64_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reduce_block = cute.ffi(name="nvshmemx_uint8_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reduce_block = cute.ffi(name="nvshmemx_uint16_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reduce_block = cute.ffi(name="nvshmemx_uint32_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reduce_block = cute.ffi(name="nvshmemx_uint64_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reduce_block = cute.ffi(name="nvshmemx_size_and_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reduce_block = cute.ffi(name="nvshmemx_uchar_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reduce_block = cute.ffi(name="nvshmemx_ushort_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reduce_block = cute.ffi(name="nvshmemx_uint_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reduce_block = cute.ffi(name="nvshmemx_ulong_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reduce_block = cute.ffi(name="nvshmemx_ulonglong_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reduce_block = cute.ffi(name="nvshmemx_int8_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reduce_block = cute.ffi(name="nvshmemx_int16_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reduce_block = cute.ffi(name="nvshmemx_int32_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reduce_block = cute.ffi(name="nvshmemx_int64_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reduce_block = cute.ffi(name="nvshmemx_uint8_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reduce_block = cute.ffi(name="nvshmemx_uint16_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reduce_block = cute.ffi(name="nvshmemx_uint32_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reduce_block = cute.ffi(name="nvshmemx_uint64_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reduce_block = cute.ffi(name="nvshmemx_size_or_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reduce_block = cute.ffi(name="nvshmemx_uchar_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reduce_block = cute.ffi(name="nvshmemx_ushort_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reduce_block = cute.ffi(name="nvshmemx_uint_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reduce_block = cute.ffi(name="nvshmemx_ulong_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reduce_block = cute.ffi(name="nvshmemx_ulonglong_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reduce_block = cute.ffi(name="nvshmemx_int8_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reduce_block = cute.ffi(name="nvshmemx_int16_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reduce_block = cute.ffi(name="nvshmemx_int32_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reduce_block = cute.ffi(name="nvshmemx_int64_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reduce_block = cute.ffi(name="nvshmemx_uint8_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reduce_block = cute.ffi(name="nvshmemx_uint16_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reduce_block = cute.ffi(name="nvshmemx_uint32_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reduce_block = cute.ffi(name="nvshmemx_uint64_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reduce_block = cute.ffi(name="nvshmemx_size_xor_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reduce_block = cute.ffi(name="nvshmemx_uchar_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reduce_block = cute.ffi(name="nvshmemx_ushort_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reduce_block = cute.ffi(name="nvshmemx_uint_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reduce_block = cute.ffi(name="nvshmemx_ulong_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reduce_block = cute.ffi(name="nvshmemx_ulonglong_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reduce_block = cute.ffi(name="nvshmemx_int8_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reduce_block = cute.ffi(name="nvshmemx_int16_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reduce_block = cute.ffi(name="nvshmemx_int32_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reduce_block = cute.ffi(name="nvshmemx_int64_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reduce_block = cute.ffi(name="nvshmemx_uint8_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reduce_block = cute.ffi(name="nvshmemx_uint16_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reduce_block = cute.ffi(name="nvshmemx_uint32_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reduce_block = cute.ffi(name="nvshmemx_uint64_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reduce_block = cute.ffi(name="nvshmemx_size_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reduce_block = cute.ffi(name="nvshmemx_char_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reduce_block = cute.ffi(name="nvshmemx_schar_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reduce_block = cute.ffi(name="nvshmemx_short_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reduce_block = cute.ffi(name="nvshmemx_int_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reduce_block = cute.ffi(name="nvshmemx_long_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reduce_block = cute.ffi(name="nvshmemx_longlong_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reduce_block = cute.ffi(name="nvshmemx_bfloat16_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reduce_block = cute.ffi(name="nvshmemx_half_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reduce_block = cute.ffi(name="nvshmemx_float_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reduce_block = cute.ffi(name="nvshmemx_double_max_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reduce_block = cute.ffi(name="nvshmemx_uchar_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reduce_block = cute.ffi(name="nvshmemx_ushort_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reduce_block = cute.ffi(name="nvshmemx_uint_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reduce_block = cute.ffi(name="nvshmemx_ulong_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reduce_block = cute.ffi(name="nvshmemx_ulonglong_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reduce_block = cute.ffi(name="nvshmemx_int8_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reduce_block = cute.ffi(name="nvshmemx_int16_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reduce_block = cute.ffi(name="nvshmemx_int32_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reduce_block = cute.ffi(name="nvshmemx_int64_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reduce_block = cute.ffi(name="nvshmemx_uint8_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reduce_block = cute.ffi(name="nvshmemx_uint16_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reduce_block = cute.ffi(name="nvshmemx_uint32_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reduce_block = cute.ffi(name="nvshmemx_uint64_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reduce_block = cute.ffi(name="nvshmemx_size_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reduce_block = cute.ffi(name="nvshmemx_char_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reduce_block = cute.ffi(name="nvshmemx_schar_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reduce_block = cute.ffi(name="nvshmemx_short_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reduce_block = cute.ffi(name="nvshmemx_int_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reduce_block = cute.ffi(name="nvshmemx_long_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reduce_block = cute.ffi(name="nvshmemx_longlong_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reduce_block = cute.ffi(name="nvshmemx_bfloat16_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reduce_block = cute.ffi(name="nvshmemx_half_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reduce_block = cute.ffi(name="nvshmemx_float_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reduce_block = cute.ffi(name="nvshmemx_double_min_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reduce_block = cute.ffi(name="nvshmemx_uchar_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reduce_block = cute.ffi(name="nvshmemx_ushort_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reduce_block = cute.ffi(name="nvshmemx_uint_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reduce_block = cute.ffi(name="nvshmemx_ulong_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reduce_block = cute.ffi(name="nvshmemx_ulonglong_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reduce_block = cute.ffi(name="nvshmemx_int8_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reduce_block = cute.ffi(name="nvshmemx_int16_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reduce_block = cute.ffi(name="nvshmemx_int32_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reduce_block = cute.ffi(name="nvshmemx_int64_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reduce_block = cute.ffi(name="nvshmemx_uint8_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reduce_block = cute.ffi(name="nvshmemx_uint16_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reduce_block = cute.ffi(name="nvshmemx_uint32_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reduce_block = cute.ffi(name="nvshmemx_uint64_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reduce_block = cute.ffi(name="nvshmemx_size_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reduce_block = cute.ffi(name="nvshmemx_char_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reduce_block = cute.ffi(name="nvshmemx_schar_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reduce_block = cute.ffi(name="nvshmemx_short_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reduce_block = cute.ffi(name="nvshmemx_int_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reduce_block = cute.ffi(name="nvshmemx_long_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reduce_block = cute.ffi(name="nvshmemx_longlong_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reduce_block = cute.ffi(name="nvshmemx_bfloat16_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reduce_block = cute.ffi(name="nvshmemx_half_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reduce_block = cute.ffi(name="nvshmemx_float_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reduce_block = cute.ffi(name="nvshmemx_double_sum_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reduce_block = cute.ffi(name="nvshmemx_uchar_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reduce_block = cute.ffi(name="nvshmemx_ushort_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reduce_block = cute.ffi(name="nvshmemx_uint_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reduce_block = cute.ffi(name="nvshmemx_ulong_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reduce_block = cute.ffi(name="nvshmemx_ulonglong_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reduce_block = cute.ffi(name="nvshmemx_int8_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reduce_block = cute.ffi(name="nvshmemx_int16_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reduce_block = cute.ffi(name="nvshmemx_int32_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reduce_block = cute.ffi(name="nvshmemx_int64_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reduce_block = cute.ffi(name="nvshmemx_uint8_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reduce_block = cute.ffi(name="nvshmemx_uint16_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reduce_block = cute.ffi(name="nvshmemx_uint32_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reduce_block = cute.ffi(name="nvshmemx_uint64_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reduce_block = cute.ffi(name="nvshmemx_size_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reduce_block = cute.ffi(name="nvshmemx_char_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reduce_block = cute.ffi(name="nvshmemx_schar_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reduce_block = cute.ffi(name="nvshmemx_short_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reduce_block = cute.ffi(name="nvshmemx_int_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reduce_block = cute.ffi(name="nvshmemx_long_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reduce_block = cute.ffi(name="nvshmemx_longlong_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reduce_block = cute.ffi(name="nvshmemx_bfloat16_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reduce_block = cute.ffi(name="nvshmemx_half_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reduce_block = cute.ffi(name="nvshmemx_float_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reduce_block = cute.ffi(name="nvshmemx_double_prod_reduce_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reducescatter_warp = cute.ffi(name="nvshmemx_uint_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reducescatter_warp = cute.ffi(name="nvshmemx_int8_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reducescatter_warp = cute.ffi(name="nvshmemx_int16_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reducescatter_warp = cute.ffi(name="nvshmemx_int32_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reducescatter_warp = cute.ffi(name="nvshmemx_int64_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reducescatter_warp = cute.ffi(name="nvshmemx_size_and_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reducescatter_warp = cute.ffi(name="nvshmemx_uint_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reducescatter_warp = cute.ffi(name="nvshmemx_int8_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reducescatter_warp = cute.ffi(name="nvshmemx_int16_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reducescatter_warp = cute.ffi(name="nvshmemx_int32_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reducescatter_warp = cute.ffi(name="nvshmemx_int64_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reducescatter_warp = cute.ffi(name="nvshmemx_size_or_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uint_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reducescatter_warp = cute.ffi(name="nvshmemx_int8_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reducescatter_warp = cute.ffi(name="nvshmemx_int16_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reducescatter_warp = cute.ffi(name="nvshmemx_int32_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reducescatter_warp = cute.ffi(name="nvshmemx_int64_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reducescatter_warp = cute.ffi(name="nvshmemx_size_xor_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reducescatter_warp = cute.ffi(name="nvshmemx_uint_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reducescatter_warp = cute.ffi(name="nvshmemx_int8_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reducescatter_warp = cute.ffi(name="nvshmemx_int16_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reducescatter_warp = cute.ffi(name="nvshmemx_int32_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reducescatter_warp = cute.ffi(name="nvshmemx_int64_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reducescatter_warp = cute.ffi(name="nvshmemx_size_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reducescatter_warp = cute.ffi(name="nvshmemx_char_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reducescatter_warp = cute.ffi(name="nvshmemx_schar_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reducescatter_warp = cute.ffi(name="nvshmemx_short_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reducescatter_warp = cute.ffi(name="nvshmemx_int_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reducescatter_warp = cute.ffi(name="nvshmemx_long_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reducescatter_warp = cute.ffi(name="nvshmemx_longlong_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reducescatter_warp = cute.ffi(name="nvshmemx_bfloat16_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reducescatter_warp = cute.ffi(name="nvshmemx_half_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reducescatter_warp = cute.ffi(name="nvshmemx_float_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reducescatter_warp = cute.ffi(name="nvshmemx_double_max_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reducescatter_warp = cute.ffi(name="nvshmemx_uint_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reducescatter_warp = cute.ffi(name="nvshmemx_int8_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reducescatter_warp = cute.ffi(name="nvshmemx_int16_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reducescatter_warp = cute.ffi(name="nvshmemx_int32_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reducescatter_warp = cute.ffi(name="nvshmemx_int64_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reducescatter_warp = cute.ffi(name="nvshmemx_size_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reducescatter_warp = cute.ffi(name="nvshmemx_char_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reducescatter_warp = cute.ffi(name="nvshmemx_schar_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reducescatter_warp = cute.ffi(name="nvshmemx_short_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reducescatter_warp = cute.ffi(name="nvshmemx_int_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reducescatter_warp = cute.ffi(name="nvshmemx_long_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reducescatter_warp = cute.ffi(name="nvshmemx_longlong_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reducescatter_warp = cute.ffi(name="nvshmemx_bfloat16_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reducescatter_warp = cute.ffi(name="nvshmemx_half_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reducescatter_warp = cute.ffi(name="nvshmemx_float_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reducescatter_warp = cute.ffi(name="nvshmemx_double_min_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uint_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reducescatter_warp = cute.ffi(name="nvshmemx_int8_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reducescatter_warp = cute.ffi(name="nvshmemx_int16_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reducescatter_warp = cute.ffi(name="nvshmemx_int32_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reducescatter_warp = cute.ffi(name="nvshmemx_int64_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reducescatter_warp = cute.ffi(name="nvshmemx_size_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reducescatter_warp = cute.ffi(name="nvshmemx_char_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reducescatter_warp = cute.ffi(name="nvshmemx_schar_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reducescatter_warp = cute.ffi(name="nvshmemx_short_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reducescatter_warp = cute.ffi(name="nvshmemx_int_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reducescatter_warp = cute.ffi(name="nvshmemx_long_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reducescatter_warp = cute.ffi(name="nvshmemx_longlong_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reducescatter_warp = cute.ffi(name="nvshmemx_bfloat16_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reducescatter_warp = cute.ffi(name="nvshmemx_half_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reducescatter_warp = cute.ffi(name="nvshmemx_float_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reducescatter_warp = cute.ffi(name="nvshmemx_double_sum_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uchar_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reducescatter_warp = cute.ffi(name="nvshmemx_ushort_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uint_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reducescatter_warp = cute.ffi(name="nvshmemx_ulong_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reducescatter_warp = cute.ffi(name="nvshmemx_ulonglong_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reducescatter_warp = cute.ffi(name="nvshmemx_int8_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reducescatter_warp = cute.ffi(name="nvshmemx_int16_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reducescatter_warp = cute.ffi(name="nvshmemx_int32_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reducescatter_warp = cute.ffi(name="nvshmemx_int64_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uint8_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uint16_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uint32_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reducescatter_warp = cute.ffi(name="nvshmemx_uint64_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reducescatter_warp = cute.ffi(name="nvshmemx_size_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reducescatter_warp = cute.ffi(name="nvshmemx_char_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reducescatter_warp = cute.ffi(name="nvshmemx_schar_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reducescatter_warp = cute.ffi(name="nvshmemx_short_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reducescatter_warp = cute.ffi(name="nvshmemx_int_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reducescatter_warp = cute.ffi(name="nvshmemx_long_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reducescatter_warp = cute.ffi(name="nvshmemx_longlong_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reducescatter_warp = cute.ffi(name="nvshmemx_bfloat16_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reducescatter_warp = cute.ffi(name="nvshmemx_half_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reducescatter_warp = cute.ffi(name="nvshmemx_float_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reducescatter_warp = cute.ffi(name="nvshmemx_double_prod_reducescatter_warp", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_and_reducescatter_block = cute.ffi(name="nvshmemx_uchar_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_and_reducescatter_block = cute.ffi(name="nvshmemx_ushort_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_and_reducescatter_block = cute.ffi(name="nvshmemx_uint_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_and_reducescatter_block = cute.ffi(name="nvshmemx_ulong_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_and_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_and_reducescatter_block = cute.ffi(name="nvshmemx_int8_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_and_reducescatter_block = cute.ffi(name="nvshmemx_int16_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_and_reducescatter_block = cute.ffi(name="nvshmemx_int32_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_and_reducescatter_block = cute.ffi(name="nvshmemx_int64_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_and_reducescatter_block = cute.ffi(name="nvshmemx_uint8_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_and_reducescatter_block = cute.ffi(name="nvshmemx_uint16_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_and_reducescatter_block = cute.ffi(name="nvshmemx_uint32_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_and_reducescatter_block = cute.ffi(name="nvshmemx_uint64_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_and_reducescatter_block = cute.ffi(name="nvshmemx_size_and_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_or_reducescatter_block = cute.ffi(name="nvshmemx_uchar_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_or_reducescatter_block = cute.ffi(name="nvshmemx_ushort_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_or_reducescatter_block = cute.ffi(name="nvshmemx_uint_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_or_reducescatter_block = cute.ffi(name="nvshmemx_ulong_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_or_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_or_reducescatter_block = cute.ffi(name="nvshmemx_int8_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_or_reducescatter_block = cute.ffi(name="nvshmemx_int16_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_or_reducescatter_block = cute.ffi(name="nvshmemx_int32_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_or_reducescatter_block = cute.ffi(name="nvshmemx_int64_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_or_reducescatter_block = cute.ffi(name="nvshmemx_uint8_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_or_reducescatter_block = cute.ffi(name="nvshmemx_uint16_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_or_reducescatter_block = cute.ffi(name="nvshmemx_uint32_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_or_reducescatter_block = cute.ffi(name="nvshmemx_uint64_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_or_reducescatter_block = cute.ffi(name="nvshmemx_size_or_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_xor_reducescatter_block = cute.ffi(name="nvshmemx_uchar_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_xor_reducescatter_block = cute.ffi(name="nvshmemx_ushort_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_xor_reducescatter_block = cute.ffi(name="nvshmemx_uint_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_xor_reducescatter_block = cute.ffi(name="nvshmemx_ulong_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_xor_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_xor_reducescatter_block = cute.ffi(name="nvshmemx_int8_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_xor_reducescatter_block = cute.ffi(name="nvshmemx_int16_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_xor_reducescatter_block = cute.ffi(name="nvshmemx_int32_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_xor_reducescatter_block = cute.ffi(name="nvshmemx_int64_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_xor_reducescatter_block = cute.ffi(name="nvshmemx_uint8_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_xor_reducescatter_block = cute.ffi(name="nvshmemx_uint16_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_xor_reducescatter_block = cute.ffi(name="nvshmemx_uint32_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_xor_reducescatter_block = cute.ffi(name="nvshmemx_uint64_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_xor_reducescatter_block = cute.ffi(name="nvshmemx_size_xor_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_max_reducescatter_block = cute.ffi(name="nvshmemx_uchar_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_max_reducescatter_block = cute.ffi(name="nvshmemx_ushort_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_max_reducescatter_block = cute.ffi(name="nvshmemx_uint_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_max_reducescatter_block = cute.ffi(name="nvshmemx_ulong_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_max_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_max_reducescatter_block = cute.ffi(name="nvshmemx_int8_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_max_reducescatter_block = cute.ffi(name="nvshmemx_int16_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_max_reducescatter_block = cute.ffi(name="nvshmemx_int32_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_max_reducescatter_block = cute.ffi(name="nvshmemx_int64_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_max_reducescatter_block = cute.ffi(name="nvshmemx_uint8_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_max_reducescatter_block = cute.ffi(name="nvshmemx_uint16_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_max_reducescatter_block = cute.ffi(name="nvshmemx_uint32_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_max_reducescatter_block = cute.ffi(name="nvshmemx_uint64_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_max_reducescatter_block = cute.ffi(name="nvshmemx_size_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_max_reducescatter_block = cute.ffi(name="nvshmemx_char_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_max_reducescatter_block = cute.ffi(name="nvshmemx_schar_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_max_reducescatter_block = cute.ffi(name="nvshmemx_short_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_max_reducescatter_block = cute.ffi(name="nvshmemx_int_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_max_reducescatter_block = cute.ffi(name="nvshmemx_long_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_max_reducescatter_block = cute.ffi(name="nvshmemx_longlong_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_max_reducescatter_block = cute.ffi(name="nvshmemx_bfloat16_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_max_reducescatter_block = cute.ffi(name="nvshmemx_half_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_max_reducescatter_block = cute.ffi(name="nvshmemx_float_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_max_reducescatter_block = cute.ffi(name="nvshmemx_double_max_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_min_reducescatter_block = cute.ffi(name="nvshmemx_uchar_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_min_reducescatter_block = cute.ffi(name="nvshmemx_ushort_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_min_reducescatter_block = cute.ffi(name="nvshmemx_uint_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_min_reducescatter_block = cute.ffi(name="nvshmemx_ulong_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_min_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_min_reducescatter_block = cute.ffi(name="nvshmemx_int8_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_min_reducescatter_block = cute.ffi(name="nvshmemx_int16_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_min_reducescatter_block = cute.ffi(name="nvshmemx_int32_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_min_reducescatter_block = cute.ffi(name="nvshmemx_int64_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_min_reducescatter_block = cute.ffi(name="nvshmemx_uint8_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_min_reducescatter_block = cute.ffi(name="nvshmemx_uint16_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_min_reducescatter_block = cute.ffi(name="nvshmemx_uint32_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_min_reducescatter_block = cute.ffi(name="nvshmemx_uint64_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_min_reducescatter_block = cute.ffi(name="nvshmemx_size_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_min_reducescatter_block = cute.ffi(name="nvshmemx_char_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_min_reducescatter_block = cute.ffi(name="nvshmemx_schar_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_min_reducescatter_block = cute.ffi(name="nvshmemx_short_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_min_reducescatter_block = cute.ffi(name="nvshmemx_int_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_min_reducescatter_block = cute.ffi(name="nvshmemx_long_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_min_reducescatter_block = cute.ffi(name="nvshmemx_longlong_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_min_reducescatter_block = cute.ffi(name="nvshmemx_bfloat16_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_min_reducescatter_block = cute.ffi(name="nvshmemx_half_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_min_reducescatter_block = cute.ffi(name="nvshmemx_float_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_min_reducescatter_block = cute.ffi(name="nvshmemx_double_min_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_sum_reducescatter_block = cute.ffi(name="nvshmemx_uchar_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_sum_reducescatter_block = cute.ffi(name="nvshmemx_ushort_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_sum_reducescatter_block = cute.ffi(name="nvshmemx_uint_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_sum_reducescatter_block = cute.ffi(name="nvshmemx_ulong_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_sum_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_sum_reducescatter_block = cute.ffi(name="nvshmemx_int8_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_sum_reducescatter_block = cute.ffi(name="nvshmemx_int16_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_sum_reducescatter_block = cute.ffi(name="nvshmemx_int32_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_sum_reducescatter_block = cute.ffi(name="nvshmemx_int64_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_sum_reducescatter_block = cute.ffi(name="nvshmemx_uint8_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_sum_reducescatter_block = cute.ffi(name="nvshmemx_uint16_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_sum_reducescatter_block = cute.ffi(name="nvshmemx_uint32_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_sum_reducescatter_block = cute.ffi(name="nvshmemx_uint64_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_sum_reducescatter_block = cute.ffi(name="nvshmemx_size_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_sum_reducescatter_block = cute.ffi(name="nvshmemx_char_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_sum_reducescatter_block = cute.ffi(name="nvshmemx_schar_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_sum_reducescatter_block = cute.ffi(name="nvshmemx_short_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_sum_reducescatter_block = cute.ffi(name="nvshmemx_int_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_sum_reducescatter_block = cute.ffi(name="nvshmemx_long_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_sum_reducescatter_block = cute.ffi(name="nvshmemx_longlong_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_sum_reducescatter_block = cute.ffi(name="nvshmemx_bfloat16_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_sum_reducescatter_block = cute.ffi(name="nvshmemx_half_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_sum_reducescatter_block = cute.ffi(name="nvshmemx_float_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_sum_reducescatter_block = cute.ffi(name="nvshmemx_double_sum_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)

uchar_prod_reducescatter_block = cute.ffi(name="nvshmemx_uchar_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

ushort_prod_reducescatter_block = cute.ffi(name="nvshmemx_ushort_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint_prod_reducescatter_block = cute.ffi(name="nvshmemx_uint_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

ulong_prod_reducescatter_block = cute.ffi(name="nvshmemx_ulong_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

ulonglong_prod_reducescatter_block = cute.ffi(name="nvshmemx_ulonglong_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

int8_prod_reducescatter_block = cute.ffi(name="nvshmemx_int8_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

int16_prod_reducescatter_block = cute.ffi(name="nvshmemx_int16_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int32_prod_reducescatter_block = cute.ffi(name="nvshmemx_int32_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

int64_prod_reducescatter_block = cute.ffi(name="nvshmemx_int64_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

uint8_prod_reducescatter_block = cute.ffi(name="nvshmemx_uint8_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint8), _CutePtrType(cutlass.Uint8), cutlass.Uint64], return_type=cutlass.Int32)

uint16_prod_reducescatter_block = cute.ffi(name="nvshmemx_uint16_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint16), _CutePtrType(cutlass.Uint16), cutlass.Uint64], return_type=cutlass.Int32)

uint32_prod_reducescatter_block = cute.ffi(name="nvshmemx_uint32_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint32), _CutePtrType(cutlass.Uint32), cutlass.Uint64], return_type=cutlass.Int32)

uint64_prod_reducescatter_block = cute.ffi(name="nvshmemx_uint64_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

size_prod_reducescatter_block = cute.ffi(name="nvshmemx_size_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Uint64), _CutePtrType(cutlass.Uint64), cutlass.Uint64], return_type=cutlass.Int32)

char_prod_reducescatter_block = cute.ffi(name="nvshmemx_char_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

schar_prod_reducescatter_block = cute.ffi(name="nvshmemx_schar_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int8), _CutePtrType(cutlass.Int8), cutlass.Uint64], return_type=cutlass.Int32)

short_prod_reducescatter_block = cute.ffi(name="nvshmemx_short_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int16), _CutePtrType(cutlass.Int16), cutlass.Uint64], return_type=cutlass.Int32)

int_prod_reducescatter_block = cute.ffi(name="nvshmemx_int_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int32), _CutePtrType(cutlass.Int32), cutlass.Uint64], return_type=cutlass.Int32)

long_prod_reducescatter_block = cute.ffi(name="nvshmemx_long_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

longlong_prod_reducescatter_block = cute.ffi(name="nvshmemx_longlong_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Int64), _CutePtrType(cutlass.Int64), cutlass.Uint64], return_type=cutlass.Int32)

bfloat16_prod_reducescatter_block = cute.ffi(name="nvshmemx_bfloat16_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.BFloat16), _CutePtrType(cutlass.BFloat16), cutlass.Uint64], return_type=cutlass.Int32)

half_prod_reducescatter_block = cute.ffi(name="nvshmemx_half_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float16), _CutePtrType(cutlass.Float16), cutlass.Uint64], return_type=cutlass.Int32)

float_prod_reducescatter_block = cute.ffi(name="nvshmemx_float_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float32), _CutePtrType(cutlass.Float32), cutlass.Uint64], return_type=cutlass.Int32)

double_prod_reducescatter_block = cute.ffi(name="nvshmemx_double_prod_reducescatter_block", params_types=[cutlass.Int32, _CutePtrType(cutlass.Float64), _CutePtrType(cutlass.Float64), cutlass.Uint64], return_type=cutlass.Int32)
