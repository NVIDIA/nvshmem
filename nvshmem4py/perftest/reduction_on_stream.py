"""
This is a Python implementation of the `reduction_on_stream` NVSHMEM Perftest

The options are identical, although CUDA graph-based kernel launches are not yet supported.
"""
import argparse

from cuda.core.experimental._event import Event
from cuda.core.experimental import Device, system
import cuda.core

import nvshmem.core

from utils import build_parser, print_runtime_options, uid_init, print_header, print_result, run_coll_benchmark

if __name__ == '__main__':
    args = build_parser()
    print_runtime_options(args)
    uid_init()
    run_coll_benchmark(args, "reduce")
    nvshmem.core.finalize()
