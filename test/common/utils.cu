/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include <unordered_map>
#include <tuple>
#include "utils.h"
#include "utils_internal.h"
#include <getopt.h>

int mype = 0, mype_node = 0, npes = 0, npes_node = 0;
int use_cubin = 0;
nvshmemBootstrapUID *bootstrap_uid = nullptr;
nvshmemBootstrapMPI *bootstrap_mpi = nullptr;

size_t _min_size = 4;
size_t _max_size = _min_size * 1024 * 1024;
size_t _min_iters = 10;
size_t _max_iters = 20;
size_t _step_factor = 2;
size_t _repeat = 1;
bool use_egm = false;
bool use_mmap = false;
size_t _mem_handle_type = MEM_TYPE_AUTO;
bool _only_p2p = false;
threadgroup_scope_t threadgroup_scope = {NVSHMEM_ALL_SCOPES, "all_scopes"};

__device__ int tile_errs_d;
int tile_errs;

// tracks mmap addr -> {user buff, size, handle}
std::unordered_map<void *, std::tuple<void *, size_t, CUmemGenericAllocationHandle>> mmaped_buffers;

void *nvml_handle = nullptr;
struct nvml_function_table nvml_ftable;
const char *env_value = nullptr;
CUmodule mymodule = NULL;

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static inline int atol_scaled(const char *str, size_t *out) {
    int scale, n;
    double p = -1.0;
    char f;
    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    } else if (p < 0) {
        return 1;
    } else
        scale = 0;

    *out = (size_t)ceil(p * (1lu << scale));
    return 0;
}

void init_cumodule(const char *str) {
    int init_error = 0;

    char exe_path[1000];
    size_t count = readlink("/proc/self/exe", exe_path, 1000);
    exe_path[count] = '\0';

    char *exe_dir = dirname(exe_path);
    char cubin_default_path[1000];
    strcpy(cubin_default_path, exe_dir);
    strcat(cubin_default_path, "/");
    strcat(cubin_default_path, str);

    // base name of the test without the extension (e.g. shmem_put_latency)
    char base_name[512];
    strncpy(base_name, str, sizeof(base_name));
    base_name[sizeof(base_name) - 1] = '\0';
    char *dot = strrchr(base_name, '.');
    if (dot && strcmp(dot, ".cubin") == 0) {
        *dot = '\0';
    }
    char cubin_bc_path[1000];
    snprintf(cubin_bc_path, sizeof(cubin_bc_path), "%s/%s_bc.cubin", exe_dir, base_name);         // produces: path/to/shmem_put_latency_bc.cubin

    char cubin_ltoir_path[1000];
    snprintf(cubin_ltoir_path, sizeof(cubin_ltoir_path), "%s/%s_ltoir.cubin", exe_dir, base_name);    // produces: path/to/shmem_put_latency_ltoir.cubin

    char* selected_path;

    if (use_cubin == NVSHMEM_CUBIN_BC) {
        selected_path = cubin_bc_path;
    } else if (use_cubin == NVSHMEM_CUBIN_LTOIR) {
        selected_path = cubin_ltoir_path;
    }

    // check if that specific .cubin with a suffix exists
    if (use_cubin == NVSHMEM_CUBIN_BC || use_cubin == NVSHMEM_CUBIN_LTOIR) {
        if (access(selected_path, R_OK) != 0) {
            fprintf(stderr,
                    "Requested NVSHMEM_TEST_CUBIN_LIBRARY=%d [0=libnvshmem.a, 1=libnvshmem_device.bc, 2=libnvshmem_device.ltoir] but cubin not found: %s\n",
                    use_cubin, selected_path);
            exit(-1);
        }
    } else {
        fprintf(stderr, "Invalid NVSHMEM_TEST_CUBIN_LIBRARY value: %d. Expected 1 or 2.\n", use_cubin);
        exit(-1);
    }

    printf("CUBIN Selected: %s\n", selected_path);
    CU_CHECK(cuModuleLoad(&mymodule, selected_path));
    init_error = nvshmemx_cumodule_init(mymodule);
    if (init_error) {
        ERROR_PRINT("cumodule_init failed \n");
        assert(false);
    }
}

void init_test_case_kernel(CUfunction *kernel, const char *kernel_name) {
    CU_CHECK(cuModuleGetFunction(kernel, mymodule, kernel_name));
}

void select_device() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(mype_node / npes_per_gpu));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    DEBUG_PRINT("mype, mype_node: %d,%d device name: %s bus id: %d \n", mype, mype_node, prop.name,
                prop.pciBusID);
}

static int parse_mode_by_env(const char *envname) {
    char *bootstrap_mode = getenv(envname);
    if (bootstrap_mode) {
        int mode = atoi(bootstrap_mode);
        if (mode < NVSHMEMTEST_USE_BOOTSTRAP_DEFAULT &&
            mode >= NVSHMEMTEST_USE_BOOTSTRAP_UNSUPPORTED)
            return NVSHMEMTEST_USE_BOOTSTRAP_UNSUPPORTED;
        else
            return mode;
    } else {
        return NVSHMEMTEST_USE_BOOTSTRAP_DEFAULT;
    }
}

nvshmemBootstrapMPI::nvshmemBootstrapMPI(int *c, char ***v)
    : nvshmemBootstrap(parse_mode_by_env("NVSHMEMTEST_USE_MPI_LAUNCHER")) {
    __attribute__((unused)) int flag = 0;
    if (usage_mode() == NVSHMEMTEST_USE_BOOTSTRAP_WITH_MPI) {
#ifdef NVSHMEMTEST_MPI_SUPPORT
        MPI_Initialized(&flag);
        if (!flag) MPI_Init(c, v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        DEBUG_PRINT("NVSHMEM: [%d of %d] MPI Bootstrap! \n", rank, nranks);
        select_device();
        /* Good to go */
#else
        throw nvshmemBootstrapMPIException("Waiving the path as missing NVSHMEMTEST_MPI_SUPPORT\n");
#endif
    } else {
        throw nvshmemBootstrapMPIException("Waiving the path with other bootstrap\n");
    }
}

nvshmemBootstrapUID::nvshmemBootstrapUID(int *c, char ***v)
    : nvshmemBootstrap(parse_mode_by_env("NVSHMEMTEST_USE_UID_BOOTSTRAP")) {
    __attribute__((unused)) int flag = 0;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int rank, nranks;
    if (usage_mode() == NVSHMEMTEST_USE_BOOTSTRAP_WITH_MPI) {
#ifdef NVSHMEMTEST_MPI_SUPPORT
        /* mpirun launcher */
        MPI_Initialized(&flag);
        if (!flag) MPI_Init(c, v);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (rank == 0) {
            nvshmemx_get_uniqueid(&id);
        }

        MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        DEBUG_PRINT("NVSHMEM: [%d of %d] UID Bootstrap! \n", rank, nranks);
        select_device();
        /* good to go */
#else
        throw nvshmemBootstrapUIDException(
            "Waiving the path as missing NVSHMEMTEST_MPI_SUPPORT \n");
#endif
    } else if (usage_mode() == NVSHMEMTEST_USE_UID_BOOTSTRAP_WITH_HYDRA) {
        /* nvshmrun.hydra launcher using PMI KeyValue Store */
        int spawned;
        SPMI_Init(&spawned);
        SPMI_Get_rank(&rank);
        SPMI_Get_size(&nranks);
        DEBUG_PRINT("PMI: [%d of %d] hello PMI key-value store! \n", rank, nranks);
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (rank == 0) {
            nvshmemx_get_uniqueid(&id);
        }

        SPMI_Bcast(&id, sizeof(nvshmemx_uniqueid_t));
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        select_device();
        /* good to go */
    } else {
        throw nvshmemBootstrapUIDException("Waiving the path with other bootstrap\n");
    }
}

template <typename T>
static void finalize_common(T &obj) {
#ifdef NVSHMEMTEST_MPI_SUPPORT
    int flag = 0;
    if (obj.usage_mode() == NVSHMEMTEST_USE_BOOTSTRAP_WITH_MPI) {
        MPI_Finalized(&flag);
        if (!flag) MPI_Finalize();
    }
#endif
}

nvshmemBootstrapMPI::~nvshmemBootstrapMPI() noexcept {
    finalize_common<nvshmemBootstrapMPI>(*this);
}

nvshmemBootstrapUID::~nvshmemBootstrapUID() noexcept {
    finalize_common<nvshmemBootstrapUID>(*this);
    if (usage_mode() == NVSHMEMTEST_USE_UID_BOOTSTRAP_WITH_HYDRA) {
        SPMI_Finalize();
    }
}

static void check_for_cumodule_tests() {
    char *test_mode = getenv("NVSHMEM_TEST_CUBIN_LIBRARY");
    if (test_mode) {
        use_cubin = atoi(test_mode);
    }
    if (use_cubin == 1) {
        printf("LLVM-IR Bitcode Library Testing Method Chosen.\n");
    } else if (use_cubin == 2) {
        printf("LTOIR Library Testing Method Chosen.\n");
    }
}

void init_wrapper(int *c, char ***v) {
    /** Only one of the bootstrap will be valid, so we can assume
      the order of trial will be MPI > UID > DEFAULT
     */
    try {
        if (nullptr == bootstrap_mpi) {
            bootstrap_mpi = new nvshmemBootstrapMPI(c, v);
        }
    } catch (nvshmemBootstrapMPIException &exp) {
        try {
            if (nullptr == bootstrap_uid) {
                bootstrap_uid = new nvshmemBootstrapUID(c, v);
            }

        } catch (nvshmemBootstrapUIDException &exp) {
            nvshmem_init();
            select_device();
            DEBUG_PRINT("end of init \n");
        }
    }

    nvshmem_barrier_all();
    check_for_cumodule_tests();
}

void read_args(int argc, char **argv) {
    int c;
    static struct option long_options[] = {{"egm", no_argument, 0, 0},
                                           {"mmap", no_argument, 0, 0},
                                           {"only_p2p", no_argument, 0, 0},
                                           {"help", no_argument, 0, 'h'},
                                           {"min_size", required_argument, 0, 'b'},
                                           {"max_size", required_argument, 0, 'e'},
                                           {"step", required_argument, 0, 'f'},
                                           {"min_iters", required_argument, 0, 'i'},
                                           {"max_iters", required_argument, 0, 'j'},
                                           {"repeat", required_argument, 0, 'r'},
                                           {"scope", required_argument, 0, 's'},
                                           {"mem_handle_type", required_argument, 0, 'm'},
                                           {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "h:b:e:f:i:j:r:s:m:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                printf(
                    "Accepted arguments: \n"
                    "-b, --min_size <minbytes> \n"
                    "-e, --max_size <maxbytes> \n"
                    "-f, --step <step factor for message sizes> \n"
                    "-i, --min_iters <min number of iterations> \n"
                    "-j, --max_iters <max number of iterations> \n"
                    "-r, --repeat <number of repetitions> \n"
                    "-s, --scope <scope> \n"
                    "--mmap (Use mmaped buffer) \n"
                    "--only_p2p (run P2P variant of test) \n"
                    "-m, --mem_handle_type <0:auto, 1:posix_fd, 2:fabric> (for mmaped buffer) \n"
                    "--egm: use EGM memory buffers \n");
                exit(0);
            case 0:
                if (strcmp(long_options[option_index].name, "egm") == 0) {
                    use_egm = true;
                } else if (strcmp(long_options[option_index].name, "mmap") == 0) {
                    use_mmap = true;
                } else if (strcmp(long_options[option_index].name, "only_p2p") == 0) {
                    _only_p2p = true;
                }
                break;
            case 'b':
                atol_scaled(optarg, &_min_size);
                break;
            case 'e':
                atol_scaled(optarg, &_max_size);
                break;
            case 'f':
                atol_scaled(optarg, &_step_factor);
                break;
            case 'i':
                atol_scaled(optarg, &_min_iters);
                break;
            case 'j':
                atol_scaled(optarg, &_max_iters);
                break;
            case 'r':
                atol_scaled(optarg, &_repeat);
                break;
            case 's':
                if (!strcmp(optarg, "thread")) {
                    threadgroup_scope.type = NVSHMEM_THREAD;
                    threadgroup_scope.name = "thread";
                } else if (!strcmp(optarg, "warp")) {
                    threadgroup_scope.type = NVSHMEM_WARP;
                    threadgroup_scope.name = "warp";
                } else if (!strcmp(optarg, "warpgroup")) {
                    threadgroup_scope.type = NVSHMEM_WARPGROUP;
                    threadgroup_scope.name = "warpgroup";
                } else if (!strcmp(optarg, "block")) {
                    threadgroup_scope.type = NVSHMEM_BLOCK;
                    threadgroup_scope.name = "block";
                } else {
                    fprintf(stderr, "Invalid scope: %s. Valid options: thread, warp, warpgroup, block\n", optarg);
                    exit(-1);
                }
                break;
            case 'm':
                atol_scaled(optarg, &_mem_handle_type);
                break;
            case '?':
                if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                return;
            default:
                abort();
        }
    }

    assert(_min_size <= _max_size);

    printf("Runtime options after parsing command line arguments\n");
    printf(
        "min_size: %zu, max_size: %zu, step_factor: %zu, min_iters: %zu, max_iters: %zu, repeat: "
        "%zu, threadgroup_scope: %s, mmap: %d, use_egm: %d, only_p2p: %d, mem_handle_type: %zu\n",
        _min_size, _max_size, _step_factor, _min_iters, _max_iters, _repeat, threadgroup_scope.name.c_str(), use_mmap, use_egm,
        _only_p2p, _mem_handle_type);
    printf(
        "Note: Above is full list of options, any given test will use only a subset of these "
        "variables.\n");
}

#define LOAD_SYM(handle, symbol, funcptr, optional, ret)        \
    do {                                                        \
        void **cast = (void **)&funcptr;                        \
        void *tmp = dlsym(handle, symbol);                      \
        *cast = tmp;                                            \
        if (*cast == NULL && !optional) {                       \
            NVSHMEMI_ERROR_PRINT("Retrieve %s failed", symbol); \
            ret = NVSHMEMX_ERROR_INTERNAL;                      \
        }                                                       \
    } while (0)

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    int status = 0;
    char path[1024];
    env_value = (const char *)getenv("NVSHMEM_CUDA_PATH");
    if (!env_value)
        snprintf(path, 1024, "%s", "libnvidia-ml.so.1");
    else
        snprintf(path, 1024, "%s/%s", env_value, "libnvidia-ml.so.1");

    *nvml_handle = dlopen(path, RTLD_NOW);
    if (!(*nvml_handle)) {
        DEBUG_PRINT("NVML library not found. %s", path);
        status = -1;
    } else {
        DEBUG_PRINT("NVML library found. %s", path);
        LOAD_SYM(*nvml_handle, "nvmlInit", nvml_ftable->nvmlInit, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlShutdown", nvml_ftable->nvmlShutdown, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetHandleByPciBusId",
                 nvml_ftable->nvmlDeviceGetHandleByPciBusId, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetP2PStatus", nvml_ftable->nvmlDeviceGetP2PStatus, 0,
                 status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetGpuFabricInfoV",
                 nvml_ftable->nvmlDeviceGetGpuFabricInfoV, 1, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetFieldValues", nvml_ftable->nvmlDeviceGetFieldValues, 0,
                    status);
    }

    if (status != 0) {
        nvshmemi_nvml_ftable_fini(nvml_ftable, nvml_handle);
    }
    return status;
}

void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    if (*nvml_handle) {
        dlclose(*nvml_handle);
        *nvml_handle = NULL;
        memset(nvml_ftable, 0, sizeof(*nvml_ftable));
    }
}

bool is_mnnvl_supported(int dev_id) {
    nvmlGpuFabricInfoV_t fabricInfo = {};
    const unsigned char zero[NVML_GPU_FABRIC_UUID_LEN] = {0};
    cudaDeviceProp prop;
    char pcie_bdf[50] = {0};
    int nbytes = 0;
    int attr;
    nvmlReturn_t nvml_status;
    nvmlDevice_t local_device;
    CUdevice my_dev;
    int cuda_drv_version;
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    CUDA_CHECK(cudaDriverGetVersion(&cuda_drv_version));
    CU_CHECK(cuDeviceGet(&my_dev, dev_id));

    /* start NVML Library */
    if (nvshmemi_nvml_ftable_init(&nvml_ftable, &nvml_handle) != 0) {
        DEBUG_PRINT("Unable to open NVML library, disabling MNNVL\n");
        return false;
    }

    nvml_status = nvml_ftable.nvmlInit();
    if (nvml_status != NVML_SUCCESS) {
        DEBUG_PRINT("Unable to initialize NVML library, disabling MNNVL\n");
        return false;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    nbytes =
        snprintf(pcie_bdf, 50, "%x:%x:%x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (nbytes < 0 || nbytes > 50) {
        DEBUG_PRINT("Unable to set device pcie bdf for our local device, disabling MNNVL\n");
        return false;
    }

    bool disable_mnnvl = false;
    const char *env_value = (const char *)getenv("NVSHMEM_DISABLE_MNNVL");
    if (env_value && (env_value[0] == '0' || env_value[0] == 'N' || env_value[0] == 'n' ||
                      env_value[0] == 'F' || env_value[0] == 'f')) {
        disable_mnnvl = false;
    } else if (env_value) {
        disable_mnnvl = true;
    }

    if (cuda_drv_version >= 12040 && prop.major >= 9 && !disable_mnnvl) {
        nvml_status = nvml_ftable.nvmlDeviceGetHandleByPciBusId(pcie_bdf, &local_device);
        if (nvml_status != NVML_SUCCESS) {
            DEBUG_PRINT("nvmlDeviceGetHandleByPciBusId failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

        /* Some platforms with older driver may not support this API, so bypass MNNVL discovery */
        if (nvml_ftable.nvmlDeviceGetGpuFabricInfoV == NULL) {
            DEBUG_PRINT("nvmlDeviceGetGpuFabricInfoV not found, MNNVL not supported\n");
            return false;
        }

        nvml_status = nvml_ftable.nvmlDeviceGetGpuFabricInfoV(local_device, &fabricInfo);
        if (nvml_status != NVML_SUCCESS) {
            DEBUG_PRINT("nvmlDeviceGetGpuFabricInfoV failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

        CU_CHECK(
            cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, my_dev));
        if (attr <= 0) {
            DEBUG_PRINT("CUDA EGM fabric not supported\n");
            return false;
        }

        if (fabricInfo.state < NVML_GPU_FABRIC_STATE_COMPLETED ||
            memcmp(fabricInfo.clusterUuid, zero, NVML_GPU_FABRIC_UUID_LEN) == 0) {
            DEBUG_PRINT("MNNVL not supported\n");
            return false;
        }
    } else {
        DEBUG_PRINT("MNNVL disabled\n");
        return false;
    }

    nvml_status = nvml_ftable.nvmlShutdown();
    if (nvml_status != NVML_SUCCESS) {
        DEBUG_PRINT("Unable to stop NVML library in NVSHMEM.");
        // is this a fatal error?
        return false;
    }
    nvshmemi_nvml_ftable_fini(&nvml_ftable, &nvml_handle);
    return true;
}

void *allocate_mmap_buffer(size_t size, int mem_handle_type, bool use_egm, bool reset_zero) {
    mype = nvshmem_my_pe();
    if (!mype) DEBUG_PRINT("allocating mmap buffer\n");
    CUmemAllocationProp prop = {};
    int dev_id, numa_id;
    size_t granularity = MEM_GRANULARITY;
    int cuda_drv_version;
    CUdevice my_dev;
    CUDA_CHECK(cudaDriverGetVersion(&cuda_drv_version));
    // Application should set the device id before calling this function
    // same as nvshmem_malloc()
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CU_CHECK(cuDeviceGet(&my_dev, dev_id));
    prop.location.id = dev_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (use_egm) {
        if (!mype) DEBUG_PRINT("using EGM memory\n");
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        CU_CHECK(cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev));
        prop.location.id = numa_id;
    } else {
        prop.allocFlags.gpuDirectRDMACapable = 1;
    }

    prop.requestedHandleTypes =
        (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if ((mem_handle_type == MEM_TYPE_AUTO) && is_mnnvl_supported(dev_id)) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    }
    // override if user specified mem handle type
    if (mem_handle_type == MEM_TYPE_FABRIC) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    } else if (mem_handle_type == MEM_TYPE_POSIX_FD) {
        prop.requestedHandleTypes =
            (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }

    // pad size to be multiple of granularity
    size = ((size + granularity - 1) / granularity) * granularity;
    if (!mype) DEBUG_PRINT("padding buffer size to %lu\n", size);
    void *bufAddr, *mmapedAddr;

    CUmemAccessDesc accessDescriptor[2];
    accessDescriptor[0].location.id = prop.location.id;
    accessDescriptor[0].location.type = prop.location.type;
    accessDescriptor[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDescriptor[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor[1].location.id = dev_id;
    accessDescriptor[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle userAllocHandle;

    CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));

    if (use_egm) {
        CU_CHECK(cuMemSetAccess((CUdeviceptr)bufAddr, size,
                                (const CUmemAccessDesc *)&accessDescriptor[0], 2));
    } else {
        CU_CHECK(cuMemSetAccess((CUdeviceptr)bufAddr, size,
                                (const CUmemAccessDesc *)&accessDescriptor[0], 1));
    }

    mmapedAddr = (void *)nvshmemx_buffer_register_symmetric(bufAddr, size, 0);
    mmaped_buffers[mmapedAddr] = std::make_tuple(bufAddr, size, userAllocHandle);

    if (reset_zero && mmapedAddr) {
        if (use_egm) {
            memset(mmapedAddr, 0, size);
        } else {
            CUDA_CHECK(cudaMemset(mmapedAddr, 0, size));
        }
    }

    return mmapedAddr;
}

size_t pad_up(size_t size) {
    return ((size + MEM_GRANULARITY - 1) / MEM_GRANULARITY) * MEM_GRANULARITY;
}

void free_mmap_buffer(void *ptr) {
    if (mmaped_buffers.count(ptr) == 0) {
        ERROR_PRINT("mmaped buffer not found\n");
        exit(1);
    }
    void *bufAddr = std::get<0>(mmaped_buffers[ptr]);
    size_t size = std::get<1>(mmaped_buffers[ptr]);
    nvshmemx_buffer_unregister_symmetric(ptr, size);
    // free the user buffer
    CU_CHECK(cuMemUnmap((CUdeviceptr)bufAddr, size));
    CU_CHECK(cuMemRelease(std::get<2>(mmaped_buffers[ptr])));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)bufAddr, size));
    mmaped_buffers.erase(ptr);
}

void finalize_wrapper() {
    if (mymodule != NULL) {
        nvshmemx_cumodule_finalize(mymodule);
        CU_CHECK(cuModuleUnload(mymodule));
    }

    nvshmem_finalize();
    if (nullptr != bootstrap_mpi) {
        delete bootstrap_mpi;
        bootstrap_mpi = nullptr;
    }

    if (nullptr != bootstrap_uid) {
        delete bootstrap_uid;
        bootstrap_uid = nullptr;
    }
}

std::string get_err_string(int errs) {
    std::string err_str;
    switch (errs) {
        case static_cast<int>(NVSHMEMX_SUCCESS):
            err_str = "Success";
            break;
        case static_cast<int>(NVSHMEMX_ERROR_INVALID_VALUE):
            err_str = "Invalid value / argument";
            break;
        case static_cast<int>(NVSHMEMX_ERROR_NOT_SUPPORTED):
            err_str = "Not supported";
            break;
        default:
            err_str = "Unknown error";
            break;
    }
    return err_str;
}
