#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <stdio.h>

template <typename T>
class nvshmemAllocation {
   public:
    nvshmemAllocation() = default;
    ~nvshmemAllocation() { dealloc(); }

    void reset(size_t capacity) {
        dealloc();
        alloc(capacity * sizeof(T));
        _capacity = capacity;
    }

    T* get() { return _data; }
    size_t size() { return _capacity; }
    void free() { dealloc(); }

   private:
    void dealloc() {
        if (_capacity) {
            nvshmem_free((void*)_data);
        }
        _capacity = 0;
    }

    void alloc(size_t size) {
        _data = (T*)nvshmem_malloc(size);
        assert(_data);
    }

    T* _data = NULL;
    size_t _capacity = 0;
};
