import pytest

from utils import uid_init, mpi_init, get_local_rank_per_node
from nvshmem.core import finalize, barrier, Teams
from cuda.core import Device


def pytest_addoption(parser):
    parser.addoption("--init-type",
                     action="store",
                     default="uid",
                     help="Method to initialize NVSHMEM",
                     choices=["uid", "mpi"])


@pytest.fixture(scope="session", autouse=True)
def nvshmem_init_fini(request):
    init_type = request.config.getoption("--init-type")
    if init_type == "uid":
        uid_init()
    elif init_type == "mpi":
        mpi_init()

    yield

    local_rank = get_local_rank_per_node()
    dev = Device(local_rank)
    dev.set_current()
    barrier(Teams.TEAM_WORLD, stream=dev.create_stream())
    dev.sync()  # Ensure all kernels are complete on this device before finalization
    finalize()
