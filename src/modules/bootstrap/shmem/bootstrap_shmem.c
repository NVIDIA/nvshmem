/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <assert.h>
#include <shmem.h>
#include <stdbool.h>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bootstrap_host_transport/env_defs_internal.h"
#include "bootstrap_util.h"
#include "env_defs.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "non_abi/nvshmemx_error.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
static size_t scratch_size;
static long *scratch;
#endif
static int nvshmem_initialized_shmem = 0;
#if defined(NVSHMEM_SHMEM_HAS_TEAMS) && defined(NVSHMEM_SHMEM_HAS_ACTIVE_SET)
static int use_legacy_collectives = 0;
#endif
enum bootstrap_shmem_mode {
    BOOTSTRAP_SHMEM_MODE_AUTO = 0,
    BOOTSTRAP_SHMEM_MODE_LEGACY,
    BOOTSTRAP_SHMEM_MODE_TEAMS
};
static enum bootstrap_shmem_mode bootstrap_shmem_mode = BOOTSTRAP_SHMEM_MODE_AUTO;
int bootstrap_debug_enable = 0;

struct nvshmemi_options_s env_attr;

/* Define common types */
#define BOOTPRI_string "\"%s\""

#define BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE) \
    if (CATEGORY == DESIRED_CAT) {                                                                 \
        switch (STYLE) {                                                                           \
            char *desc_wrapped;                                                                    \
            case BOOTSTRAP_OPTIONS_STYLE_INFO:                                                     \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, "\t", 1);                \
                printf("  NVSHMEM_%-20s " BOOTPRI_##KIND " (type: %s, default: " BOOTPRI_##KIND    \
                       ")\n\t%s\n",                                                                \
                       #NAME, NVSHFMT_##KIND(env_attr.NAME), #KIND, NVSHFMT_##KIND(DEFAULT),       \
                       desc_wrapped);                                                              \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            case BOOTSTRAP_OPTIONS_STYLE_RST:                                                      \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, NULL, 0);                \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                          \
                printf("\n");                                                                      \
                printf("| *Type: %s*\n", #KIND);                                                   \
                printf("| *Default: " BOOTPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));              \
                printf("\n");                                                                      \
                printf("%s\n", desc_wrapped);                                                      \
                printf("\n");                                                                      \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            default:                                                                               \
                assert(0);                                                                         \
        }                                                                                          \
    }

static int bootstrap_shmem_showinfo(struct bootstrap_handle *handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");
    return 0;
}

void bootstrap_shmem_global_exit(int status) { shmem_global_exit(status); }

static int parse_bootstrap_shmem_mode(enum bootstrap_shmem_mode *mode) {
    const char *value = env_attr.BOOTSTRAP_SHMEM_MODE;

    if (!value || !value[0] || !strcasecmp(value, "auto")) {
        *mode = BOOTSTRAP_SHMEM_MODE_AUTO;
        return 0;
    }

    if (!strcasecmp(value, "legacy")) {
        *mode = BOOTSTRAP_SHMEM_MODE_LEGACY;
        return 0;
    }

    if (!strcasecmp(value, "teams")) {
        *mode = BOOTSTRAP_SHMEM_MODE_TEAMS;
        return 0;
    }

    BOOTSTRAP_ERROR_PRINT("Invalid NVSHMEM_BOOTSTRAP_SHMEM_MODE='%s'. "
                          "Allowed values: auto, legacy, teams.\n",
                          value);
    return NVSHMEMX_ERROR_INTERNAL;
}

#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
/*
 * Round length up to the nearest multiple of 4 bytes. shmem_collect32 and
 * shmem_alltoall32 operate on 32-bit elements, so sub-4-byte payloads
 * (e.g. bool) must be padded to avoid truncation.
 */
static inline size_t round_up_4(int length) { return (size_t)((length + 3) & ~3); }

/*
 * Legacy active-set collectives bootstrap path (OpenSHMEM < 1.5).
 *
 * These APIs are deprecated in the OpenSHMEM specification. This fallback
 * is also used at runtime when teams-based collectives are detected at
 * compile time but fail at runtime (e.g. OpenMPI oshmem).
 */

static int bootstrap_shmem_barrier_legacy(struct bootstrap_handle *handle) {
    shmem_barrier_all();
    return 0;
}

static int bootstrap_shmem_allgather_legacy(const void *sendbuf, void *recvbuf, int length,
                                            struct bootstrap_handle *handle) {
    int status = 0;
    size_t length_rup = round_up_4(length);
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc(length_rup);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc(length_rup * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    shmem_barrier_all();

    memset(sendbuf_i, 0, length_rup);
    memcpy(sendbuf_i, sendbuf, length);

    shmem_barrier_all();
    assert(scratch_size >= SHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    shmem_collect32(recvbuf_i, sendbuf_i, length_rup / 4, 0, 0, handle->pg_size, scratch);
    shmem_barrier_all();

    for (int i = 0; i < handle->pg_size; i++) {
        memcpy((char *)recvbuf + (size_t)i * length,
               (char *)recvbuf_i + (size_t)i * length_rup, length);
    }

out:
    if (!status && (sendbuf_i || recvbuf_i)) shmem_barrier_all();
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    if (!status && (sendbuf_i || recvbuf_i)) shmem_barrier_all();
    return status;
}

static int bootstrap_shmem_alltoall_legacy(const void *sendbuf, void *recvbuf, int length,
                                           struct bootstrap_handle *handle) {
    int status = 0;
    size_t length_rup = round_up_4(length);
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc(length_rup * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc(length_rup * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    shmem_barrier_all();

    memset(sendbuf_i, 0, length_rup * handle->pg_size);
    for (int i = 0; i < handle->pg_size; i++) {
        memcpy((char *)sendbuf_i + (size_t)i * length_rup,
               (const char *)sendbuf + (size_t)i * length, length);
    }

    shmem_barrier_all();
    assert(scratch_size >= SHMEM_ALLTOALL_SYNC_SIZE * sizeof(long));
    shmem_alltoall32(recvbuf_i, sendbuf_i, length_rup / 4, 0, 0, handle->pg_size, scratch);
    shmem_barrier_all();

    for (int i = 0; i < handle->pg_size; i++) {
        memcpy((char *)recvbuf + (size_t)i * length,
               (char *)recvbuf_i + (size_t)i * length_rup, length);
    }

out:
    if (!status && (sendbuf_i || recvbuf_i)) shmem_barrier_all();
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    if (!status && (sendbuf_i || recvbuf_i)) shmem_barrier_all();
    return status;
}

#endif /* NVSHMEM_SHMEM_HAS_ACTIVE_SET */

#ifdef NVSHMEM_SHMEM_HAS_TEAMS
/*
 * OpenSHMEM >= 1.5 teams-based bootstrap path.
 *
 * Uses shmem_uint8_fcollect / shmem_uint8_alltoall on SHMEM_TEAM_WORLD.
 * These operate on bytes (uint8_t), so no 4-byte round-up is needed.
 * No scratch/pSync buffer required — the teams API manages synchronization
 * internally.
 *
 * If a teams collective fails at runtime (e.g. OpenMPI oshmem returns
 * ERR_NOT_IMPLEMENTED), we fall back to the legacy active-set path for
 * all subsequent calls (if available).
 */

#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
static void fallback_to_legacy(struct bootstrap_handle *handle, int rc) {
    if (bootstrap_shmem_mode == BOOTSTRAP_SHMEM_MODE_TEAMS) {
        BOOTSTRAP_ERROR_PRINT("OpenSHMEM teams collective failed at runtime (rc=%d) while "
                              "NVSHMEM_BOOTSTRAP_SHMEM_MODE=teams is set.\n",
                              rc);
        return;
    }

    if (!use_legacy_collectives) {
        use_legacy_collectives = 1;
        fprintf(stderr,
                "NVSHMEM: WARNING: OpenSHMEM teams-based collective failed at runtime. "
                "Falling back to deprecated active-set collectives. This can happen with "
                "OpenMPI oshmem which defines teams symbols but does not implement them.\n");
        handle->allgather = bootstrap_shmem_allgather_legacy;
        handle->alltoall = bootstrap_shmem_alltoall_legacy;
        handle->barrier = bootstrap_shmem_barrier_legacy;
    }
}
#define TEAMS_FALLBACK(handle, func, ...) \
    do { \
        fallback_to_legacy(handle, rc); \
        if (bootstrap_shmem_mode == BOOTSTRAP_SHMEM_MODE_TEAMS) return NVSHMEMX_ERROR_INTERNAL; \
        return func(__VA_ARGS__); \
    } while (0)
#else
#define TEAMS_FALLBACK(handle, func, ...) \
    do { \
        BOOTSTRAP_ERROR_PRINT("shmem teams collective failed (rc=%d) and legacy " \
                              "active-set fallback is not available.\n", rc); \
        return NVSHMEMX_ERROR_INTERNAL; \
    } while (0)
#endif

static int bootstrap_shmem_barrier(struct bootstrap_handle *handle) {
    int rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    if (rc != 0) {
        TEAMS_FALLBACK(handle, bootstrap_shmem_barrier_legacy, handle);
    }
    return 0;
}

static int bootstrap_shmem_allgather(const void *sendbuf, void *recvbuf, int length,
                                     struct bootstrap_handle *handle) {
    int status = 0;
    int rc;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc(length);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc((size_t)length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");

    memcpy(sendbuf_i, sendbuf, length);

    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    if (rc != 0) goto fallback;
    rc = shmem_uint8_fcollect(SHMEM_TEAM_WORLD, recvbuf_i, sendbuf_i, length);
    if (rc != 0) goto fallback;
    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    if (rc != 0) goto fallback;

    memcpy(recvbuf, recvbuf_i, (size_t)length * handle->pg_size);
    goto out;

fallback:
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    TEAMS_FALLBACK(handle, bootstrap_shmem_allgather_legacy, sendbuf, recvbuf, length, handle);

out:
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    return status;
}

static int bootstrap_shmem_alltoall(const void *sendbuf, void *recvbuf, int length,
                                    struct bootstrap_handle *handle) {
    int status = 0;
    int rc;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc((size_t)length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc((size_t)length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");

    memcpy(sendbuf_i, sendbuf, (size_t)length * handle->pg_size);

    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    if (rc != 0) goto fallback;
    rc = shmem_uint8_alltoall(SHMEM_TEAM_WORLD, recvbuf_i, sendbuf_i, length);
    if (rc != 0) goto fallback;
    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    if (rc != 0) goto fallback;

    memcpy(recvbuf, recvbuf_i, (size_t)length * handle->pg_size);
    goto out;

fallback:
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    TEAMS_FALLBACK(handle, bootstrap_shmem_alltoall_legacy, sendbuf, recvbuf, length, handle);

out:
    if (sendbuf_i) shmem_free(sendbuf_i);
    if (recvbuf_i) shmem_free(recvbuf_i);
    return status;
}

#endif /* NVSHMEM_SHMEM_HAS_TEAMS */

static int bootstrap_shmem_finalize(bootstrap_handle_t *handle) {
    if (nvshmem_initialized_shmem) {
#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
        if (scratch) shmem_free(scratch);
#endif
        shmem_finalize();
    }
    return 0;
}

#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
static int init_legacy_scratch(struct bootstrap_handle *handle) {
    int status = 0;

    scratch_size = MAX(SHMEM_COLLECT_SYNC_SIZE, SHMEM_ALLTOALL_SYNC_SIZE) * sizeof(long);
    scratch = shmem_malloc(scratch_size);
    BOOTSTRAP_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");

    for (size_t i = 0; i < scratch_size / sizeof(long); ++i) {
        scratch[i] = SHMEM_SYNC_VALUE;
    }

out:
    return status;
}
#endif

int nvshmemi_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle,
                                   const int nvshmem_version) {
    int status = 0;
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, nvshmem_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "SHMEM bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, nvshmem_version);
        exit(-1);
    }

    nvshmemi_env_options_init(&env_attr);
    status = parse_bootstrap_shmem_mode(&bootstrap_shmem_mode);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to parse NVSHMEM_BOOTSTRAP_SHMEM_MODE\n");
    if (arg == NULL || *(int *)arg) {
        shmem_init();
        nvshmem_initialized_shmem = 1;
    }

    handle->pg_rank = shmem_my_pe();
    handle->pg_size = shmem_n_pes();

#ifdef NVSHMEM_SHMEM_HAS_ACTIVE_SET
    /* Initialize legacy scratch buffer (also used as fallback for teams path) */
    status = init_legacy_scratch(handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to initialize legacy scratch buffer\n");
#endif

#if defined(NVSHMEM_SHMEM_HAS_TEAMS) && defined(NVSHMEM_SHMEM_HAS_ACTIVE_SET)
    if (bootstrap_shmem_mode == BOOTSTRAP_SHMEM_MODE_LEGACY) {
        use_legacy_collectives = 1;
        if (handle->pg_rank == 0) {
            fprintf(stderr,
                    "NVSHMEM: WARNING: Using deprecated OpenSHMEM active-set collectives for "
                    "bootstrap because NVSHMEM_BOOTSTRAP_SHMEM_MODE=legacy.\n");
        }
        handle->allgather = bootstrap_shmem_allgather_legacy;
        handle->alltoall = bootstrap_shmem_alltoall_legacy;
        handle->barrier = bootstrap_shmem_barrier_legacy;
    } else {
        handle->allgather = bootstrap_shmem_allgather;
        handle->alltoall = bootstrap_shmem_alltoall;
        handle->barrier = bootstrap_shmem_barrier;
    }
#elif defined(NVSHMEM_SHMEM_HAS_TEAMS)
    if (bootstrap_shmem_mode == BOOTSTRAP_SHMEM_MODE_LEGACY) {
        BOOTSTRAP_ERROR_PRINT("NVSHMEM_BOOTSTRAP_SHMEM_MODE=legacy was requested, but this "
                              "OpenSHMEM build does not provide legacy active-set collectives.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    handle->allgather = bootstrap_shmem_allgather;
    handle->alltoall = bootstrap_shmem_alltoall;
    handle->barrier = bootstrap_shmem_barrier;
#elif defined(NVSHMEM_SHMEM_HAS_ACTIVE_SET)
    if (bootstrap_shmem_mode == BOOTSTRAP_SHMEM_MODE_TEAMS) {
        BOOTSTRAP_ERROR_PRINT("NVSHMEM_BOOTSTRAP_SHMEM_MODE=teams was requested, but this "
                              "OpenSHMEM build does not provide teams collectives.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    if (handle->pg_rank == 0) {
        fprintf(stderr,
                "NVSHMEM: WARNING: Using deprecated OpenSHMEM active-set collectives for "
                "bootstrap. Upgrade to OpenSHMEM >= 1.5 for teams-based bootstrap support.\n");
    }
    handle->allgather = bootstrap_shmem_allgather_legacy;
    handle->alltoall = bootstrap_shmem_alltoall_legacy;
    handle->barrier = bootstrap_shmem_barrier_legacy;
#else
#error "Neither NVSHMEM_SHMEM_HAS_TEAMS nor NVSHMEM_SHMEM_HAS_ACTIVE_SET is defined"
#endif

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(nvshmem_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? nvshmem_version
                          : bootstrap_version;
    handle->global_exit = bootstrap_shmem_global_exit;
    handle->finalize = bootstrap_shmem_finalize;
    handle->show_info = bootstrap_shmem_showinfo;
    handle->comm_state = NULL;
    handle->pre_init_ops = NULL;

out:
    return status;
}
