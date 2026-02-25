import nvshmem

import os


def test_import_modules():
    print("Testing import modules")
    # Import modules
    import nvshmem
    import nvshmem.core
    import nvshmem.bindings

    # Import core APIs
    from nvshmem.core import init, finalize

    # Import bindings
    print("Testing import bindings")
    attr = nvshmem.bindings.InitAttr()

    # Can't run these because it assumes stuff about nvshmem state
    from nvshmem.bindings import hostlib_finalize, hostlib_init_attr, uniqueid, check_status

    # Get version info
    print(nvshmem.core.get_version())


if __name__ == '__main__':
    test_import_modules()
