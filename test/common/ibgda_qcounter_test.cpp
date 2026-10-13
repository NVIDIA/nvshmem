/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Provider checks must remain active in Release builds as well.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <cstdio>
#include "../../src/modules/transport/ibgda/ibgda_qcounter.hpp"

namespace qc = nvshmemt_ibgda_qcounter;
struct fake_provider {
    int created = 0, queried = 0, destroyed = 0;
    bool create_fail = false, query_fail = false, destroy_fail = false;
    unsigned char id = 7, create_status = 0, query_status = 0;
};
static fake_provider *current;
static void *create(void *context, const void *in, size_t in_size, void *out, size_t out_size) {
    auto *provider = static_cast<fake_provider *>(context);
    assert(provider == current && in_size == 16 && out_size == 16);
    auto *bytes = static_cast<const unsigned char *>(in);
    assert(bytes[0] == 7 && bytes[1] == 0x71);
    for (size_t i = 2; i < in_size; ++i) assert(bytes[i] == 0);
    ++provider->created;
    auto *response = static_cast<unsigned char *>(out);
    response[0] = provider->create_status;
    response[11] = provider->id;
    if (provider->create_fail) { errno = EACCES; return nullptr; }
    return provider;
}
static int query(void *object, const void *in, size_t in_size, void *out, size_t out_size) {
    assert(object == current && in_size == 32 && out_size == 256);
    auto *bytes = static_cast<const unsigned char *>(in);
    assert(bytes[0] == 7 && bytes[1] == 0x73 && bytes[31] == current->id);
    // In particular, other_vport, clear and aggregate must remain zero.
    for (size_t i = 2; i < 31; ++i) assert(bytes[i] == 0);
    ++current->queried;
    auto *response = static_cast<unsigned char *>(out);
    response[0] = current->query_status;
    for (size_t i = 0; i < 11; ++i) response[16 + 8 * i + 3] = static_cast<unsigned char>(i + 1);
    response[16] = 0xff; response[17] = 0xff; response[18] = 0xff; response[19] = 0xff;
    if (current->query_fail) { errno = EIO; return -1; }
    return 0;
}
static int destroy(void *object) {
    assert(object == current);
    ++current->destroyed;
    if (current->destroy_fail) { errno = EIO; return -1; }
    return 0;
}
int main() {
    qc::operations ops = {create, query, destroy};
    int cases = 0;
    {
        fake_provider provider; current = &provider;
        qc::resource counter(false, {8, true, true}, current, ops);
        assert(counter.availability() == qc::state::disabled && !counter.query(1).valid);
        assert(counter.close() && counter.close() && !provider.created && !provider.queried && !provider.destroyed); ++cases;
    }
    {
        fake_provider provider; current = &provider;
        qc::resource counter(true, {0, true, true}, current, ops);
        assert(counter.availability() == qc::state::capability_unavailable);
        assert(!counter.query(1).valid && counter.close() && !provider.created); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.create_fail = true; provider.create_status = 2;
        qc::resource counter(true, {8, true, true}, current, ops);
        assert(counter.availability() == qc::state::allocation_failed && counter.error() == EACCES && counter.firmware_status() == 2);
        assert(!counter.query(1).valid && counter.close() && !provider.destroyed); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.id = 0;
        qc::resource counter(true, {8, true, true}, current, ops);
        assert(counter.availability() == qc::state::reserved_zero_id && counter.id() == 0);
        assert(!counter.query(1).valid && !provider.queried && counter.close() && provider.destroyed == 1); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.create_status = 2;
        qc::resource counter(true, {8, true, true}, current, ops);
        assert(counter.availability() == qc::state::allocation_failed && !counter.query(1).valid);
        assert(counter.close() && provider.destroyed == 1); ++cases;
    }
    {
        fake_provider provider; current = &provider;
        qc::resource counter(true, {8, true, true}, current, ops);
        assert(counter.id() == 7 && !counter.query(1).valid && !provider.queried);
        counter.attached(false);
        assert(!counter.query(2).valid && !counter.close() && !provider.destroyed);
        assert(counter.qp_destroyed(false) && counter.close() && provider.destroyed == 1); ++cases;
    }
    {
        fake_provider provider; current = &provider;
        qc::resource counter(true, {8, true, true}, current, ops);
        counter.attached(true); counter.attached(false);
        auto sample = counter.query(1);
        assert(sample.valid && sample.values[0] == UINT32_MAX && sample.values[5] == 6 && sample.values[10] == 11);
        assert(!counter.close() && !provider.destroyed);
        assert(counter.qp_destroyed(true) && !counter.query(2).valid);
        assert(counter.qp_destroyed(false) && !counter.qp_destroyed(false));
        assert(counter.close() && counter.close() && provider.created == 1 && provider.destroyed == 1);
        assert(!counter.query(3).valid && provider.queried == 1); ++cases;
    }
    {
        fake_provider provider; current = &provider;
        qc::resource counter(true, {8, false, false}, current, ops); counter.attached(true);
        auto sample = counter.query(0, 1000);
        assert(sample.valid && sample.supported[0] && !sample.supported[5] && !sample.supported[10]);
        assert(!counter.query(999, 1000).valid && provider.queried == 1);
        assert(counter.query(1000, 1000).valid && provider.queried == 2);
        assert(counter.qp_destroyed(true) && counter.close()); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.query_fail = true;
        qc::resource counter(true, {8, true, true}, current, ops); counter.attached(true);
        auto sample = counter.query(1); assert(!sample.valid && sample.error == EIO);
        assert(counter.qp_destroyed(true) && counter.close() && provider.destroyed == 1); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.query_status = 2;
        qc::resource counter(true, {8, true, true}, current, ops); counter.attached(true);
        auto sample = counter.query(1); assert(!sample.valid && sample.firmware_status == 2);
        assert(counter.qp_destroyed(true) && counter.close()); ++cases;
    }
    {
        fake_provider provider; current = &provider; provider.destroy_fail = true;
        qc::resource counter(true, {8, true, true}, current, ops);
        assert(!counter.close() && !counter.close() && provider.destroyed == 1);
        assert(counter.availability() == qc::state::destroy_failed && !counter.query(1).valid); ++cases;
    }
    std::printf("{\"status\":\"PASS\",\"provider_lifecycle_cases\":%d,\"hardware_test\":false}\n", cases);
}
