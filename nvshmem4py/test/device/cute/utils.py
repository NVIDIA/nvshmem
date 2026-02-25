import importlib.util
from pathlib import Path

_UTILS_PATH = Path(__file__).resolve().parents[2] / "utils.py"
_spec = importlib.util.spec_from_file_location("_nvshmem_test_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

uid_init = _utils.uid_init
mpi_init = _utils.mpi_init
get_local_rank_per_node = _utils.get_local_rank_per_node

__all__ = ["uid_init", "mpi_init", "get_local_rank_per_node"]
