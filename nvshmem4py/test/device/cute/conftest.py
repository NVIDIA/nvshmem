import pytest

from utils import uid_init, mpi_init, get_local_rank_per_node
from nvshmem.core import finalize
from cuda.core.experimental import Device
from test_device_rma import _finalize_kernels


def pytest_addoption(parser):
    parser.addoption("--init-type", action="store", default="uid", help="Method to initialize NVSHMEM", choices=["uid", "mpi"])


@pytest.fixture(scope="function", autouse=True)
def nvshmem_init_fini(request):
    init_type = request.config.getoption("--init-type")
    if init_type == "uid":
        uid_init()
    elif init_type == "mpi":
        mpi_init()

    yield

    # Ensure the correct device context is current before finalize.
    local_rank = get_local_rank_per_node()
    Device(local_rank).set_current()
    _finalize_kernels()
    finalize()
