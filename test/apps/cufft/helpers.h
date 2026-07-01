/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// This file should not depend on anything cuFFT-related / MPI / CUDA and should only rely on C++ +
// STL
#ifndef __TEST_CUFFTMPI_HELPERS_H__
#define __TEST_CUFFTMPI_HELPERS_H__

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

void ffprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    fflush(stdout);
    va_end(args);
}

void printf_date(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm *local = std::localtime(&time);

    printf("%d-%d-%dT%d:%d:%d\n", local->tm_year, local->tm_mon + 1, local->tm_mday, local->tm_hour,
           local->tm_min, local->tm_sec);
}
template <typename T>
struct statistics {
    T average, median, stdev, stdev_rel, pctl10, pctl90;
};

template <typename T>
statistics<T> compute_statistics(const std::vector<T> &v) {
    assert(v.size() > 0);

    std::vector<T> w = v;
    std::sort(w.begin(), w.end());
    statistics<T> stats;

    stats.median = w[w.size() / 2];

    T sum = std::accumulate(v.begin(), v.end(), 0.0);
    stats.average = sum / v.size();

    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    stats.stdev = std::sqrt(sq_sum / v.size() - stats.average * stats.average);

    stats.stdev_rel = stats.stdev / stats.average;

    stats.pctl10 = w[std::floor(w.size() * 0.1)];     // 10th percentile
    stats.pctl90 = w[std::ceil(w.size() * 0.9) - 1];  // 90th percentile

    return stats;
}

// A class similar to std::optional but without the need for C++17

template <typename T>
struct optional {
    bool has_value() { return set; }
    T value() { return val; }
    T value_or(T v) { return (this->set ? val : v); }
    optional(T v) : set(true), val(v) {};
    optional() : set(false) {};

   private:
    T val;
    const bool set;
};

template <typename T>
optional<T> make_optional(T v) {
    return optional<T>(v);
}

struct empty_optional {
    template <typename T>
    operator optional<T>() const {
        return optional<T>();
    }
};

static const empty_optional nullopt;

#endif  // __TEST_CUFFTMPI_HELPERS_H__
