#include <gtest/gtest.h>

#include "non_abi/nvshmemx_error.h"

TEST(nvshmemx_status_string, KnownValuesMatchEnumeratorNames) {
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_SUCCESS), "NVSHMEMX_SUCCESS");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_INVALID_VALUE),
                 "NVSHMEMX_ERROR_INVALID_VALUE");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_OUT_OF_MEMORY),
                 "NVSHMEMX_ERROR_OUT_OF_MEMORY");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_NOT_SUPPORTED),
                 "NVSHMEMX_ERROR_NOT_SUPPORTED");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_SYMMETRY), "NVSHMEMX_ERROR_SYMMETRY");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_GPU_NOT_SELECTED),
                 "NVSHMEMX_ERROR_GPU_NOT_SELECTED");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED),
                 "NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED");
    EXPECT_STREQ(nvshmemx_status_string(NVSHMEMX_ERROR_INTERNAL), "NVSHMEMX_ERROR_INTERNAL");
}

TEST(nvshmemx_status_string, UnknownValuesReturnUnknownString) {
    EXPECT_STREQ(nvshmemx_status_string(-1), "NVSHMEMX_ERROR_<unknown>");
    EXPECT_STREQ(nvshmemx_status_string(999999), "NVSHMEMX_ERROR_<unknown>");
}
