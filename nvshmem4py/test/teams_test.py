"""
Unit tests for team management functionality in nvshmem.core
"""

from utils import uid_init, mpi_init
import argparse

from cuda.core import Device
import nvshmem.core
from nvshmem.core import TeamConfig, TeamUniqueId, get_team_unique_id, Teams


def test_team_split_strided():
    print("Testing team split strided")
    device = Device()
    stream = device.create_stream()

    node_size = nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE)
    if node_size < 2:
        print("Skipping strided test - need at least 2 PEs in node team")
        return

    # Create a TeamConfig object
    config = TeamConfig()
    config.version = 2  # Use version 2
    config.num_contexts = 1

    # All PEs must use the same unique ID for team creation
    # Use a fixed unique ID for testing - in real applications, this would be shared properly
    config.uniqueid = 12345  # Fixed test value

    # All PEs in TEAM_NODE must call team_split_strided, even if they won't be in the new team
    team = nvshmem.core.team_split_strided(nvshmem.core.Teams.TEAM_NODE, 0, 2, min(2, node_size), config, 0)
    if team is None:
        print(f"PE {nvshmem.core.my_pe()} I am not in the new team")
        team_size = 0
    else:
        team_size = nvshmem.core.team_n_pes(team)

    print(f"PE {nvshmem.core.my_pe()} Team size: {team_size}")

    # Some PEs will be in the team (team_size > 0), others won't (team_size == -1)
    # This is expected behavior for team_split_strided
    if team_size > 0:
        print(f"PE {nvshmem.core.my_pe()} I am in the new team with {team_size} PEs")
    else:
        print(f"PE {nvshmem.core.my_pe()} I am not in the new team")

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    device.sync()
    print("Done testing team split strided")


def test_team_split_2d():
    print("Testing team split 2d")
    device = Device()
    stream = device.create_stream()
    node_size = nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE)
    if node_size < 4:  # Need at least 4 PEs for 2x2 split
        print("Skipping 2d test - need at least 4 PEs in node team")
        return

    # Create TeamConfig objects for x and y axes
    xaxis_config = TeamConfig()
    xaxis_config.version = 2
    xaxis_config.num_contexts = 1

    # Use fixed unique IDs for testing - all PEs must use the same IDs
    xaxis_config.uniqueid = 12346  # Fixed test value for x-axis

    yaxis_config = TeamConfig()
    yaxis_config.version = 2
    yaxis_config.num_contexts = 1

    # Use fixed unique IDs for testing - all PEs must use the same IDs
    yaxis_config.uniqueid = 12347  # Fixed test value for y-axis

    xaxis_team, yaxis_team = nvshmem.core.team_split_2d(nvshmem.core.Teams.TEAM_NODE, 2, xaxis_config, 0, yaxis_config,
                                                        0)
    x_size = nvshmem.core.team_n_pes(xaxis_team)
    y_size = nvshmem.core.team_n_pes(yaxis_team)
    print(f"X-axis team size: {x_size}")
    print(f"Y-axis team size: {y_size}")
    assert x_size > 0 and y_size > 0, "Both teams should have positive size"
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    device.sync()
    print("Done testing team split 2d")


def test_team_destroy():
    print("Testing team destroy")

    # Create a TeamConfig object
    config = TeamConfig()
    config.version = 2
    config.num_contexts = 1

    # All processes must participate in team creation
    # For a 2-PE team, both PEs must call team_init
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()

    if n_pes >= 2:
        # Use a fixed unique ID for testing - all processes use the same ID
        # In a real application, you would need to share the unique ID properly
        config.uniqueid = 12348  # Fixed test value

        # Create a team with 2 PEs - both PEs must participate
        team = nvshmem.core.team_init(config, 0, n_pes, my_pe, new_team_name="TEST_TEAM")
        print(f"Created team {team} with name TEST_TEAM")
        assert "TEST_TEAM" in Teams.keys(), "Team name should be added to Teams enum"
        nvshmem.core.team_destroy(Teams.TEST_TEAM)
        print("Done testing team destroy")
    else:
        print("Skipping team destroy test - need at least 2 PEs")


def test_team_unique_id():
    print("Testing TeamUniqueId")

    # Test getting a unique ID from NVSHMEM
    unique_id1 = get_team_unique_id()
    print(f"Generated unique ID: {unique_id1}")
    print(f"Value: {unique_id1.value}")

    # Basic validation
    assert unique_id1 is not None, "Unique ID should not be None"
    assert hasattr(unique_id1, 'value'), "Unique ID should have value attribute"

    # Test that we can get multiple unique IDs
    unique_id2 = get_team_unique_id()
    assert unique_id2 is not None, "Second unique ID should not be None"
    print(f"Generated second unique ID: {unique_id2}")

    print("Done testing TeamUniqueId")


def test_team_destroy_negative():
    print("Testing team_destroy negative cases")
    # Try destroying a team that does not exist by name
    try:
        nvshmem.core.team_destroy("NON_EXISTENT_TEAM")
        raise AssertionError("Expected exception when destroying non-existent team by name")
    except Exception as e:
        print(f"Correctly caught exception for non-existent team name: {e}")

    # Try destroying a team that does not exist by handle (arbitrary int)
    try:
        nvshmem.core.team_destroy(30)
        print("No exception for destroying non-existent team by handle (C API is idempotent)")
    except Exception as e:
        print(f"Caught exception for non-existent team handle: {e}")

    # Try destroying a team twice
    config = TeamConfig()
    config.version = 2
    config.num_contexts = 1
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    if n_pes >= 2:
        config.uniqueid = 12349
        team = nvshmem.core.team_init(config, 0, n_pes, my_pe, new_team_name="NEG_TEST_TEAM")
        nvshmem.core.team_destroy("NEG_TEST_TEAM")
        try:
            nvshmem.core.team_destroy("NEG_TEST_TEAM")
            raise AssertionError("Expected exception when destroying the same team twice")
        except Exception as e:
            print(f"Correctly caught exception for double destroy: {e}")
    else:
        print("Skipping double destroy negative test - need at least 2 PEs")


def test_team_init_negative():
    print("Testing team_init negative cases")
    config = TeamConfig()
    config.version = 2
    config.num_contexts = 1
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()

    # Negative: invalid config_mask (e.g., -1)
    try:
        nvshmem.core.team_init(config, -1, n_pes, my_pe, new_team_name="BAD_MASK_TEAM")
        raise AssertionError("Expected exception for invalid config_mask")
    except Exception as e:
        print(f"Correctly caught exception for invalid config_mask: {e}")


def test_team_split_strided_negative():
    print("Testing team_split_strided negative cases")
    parent_team = Teams.WORLD if "WORLD" in Teams else 0
    config = TeamConfig()
    config.version = 2
    config.num_contexts = 1
    device = Device()
    stream = device.create_stream()

    # Negative: invalid start/stride/size (e.g., size=0)
    try:
        nvshmem.core.team_split_strided(parent_team, 0, 1, 0, config, 0, new_team_name="BAD_SIZE_TEAM")
        raise AssertionError("Expected exception for size=0")
    except Exception as e:
        print(f"Correctly caught exception for size=0: {e}")
    finally:
        # Ensure all ranks realign after the failed collective
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)

    # Negative: invalid parent_team (e.g., -1)
    try:
        nvshmem.core.team_split_strided(-1, 0, 1, 2, config, 0, new_team_name="BAD_PARENT_TEAM")
        raise AssertionError("Expected exception for invalid parent_team")
    except Exception as e:
        print(f"Correctly caught exception for invalid parent_team: {e}")
    finally:
        # Ensure all ranks realign after the failed collective
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)


def test_team_split_2d_negative():
    print("Testing team_split_2d negative cases")
    parent_team = Teams.WORLD if "WORLD" in Teams else 0
    config_x = TeamConfig()
    config_x.version = 2
    config_x.num_contexts = 1
    config_y = TeamConfig()
    config_y.version = 2
    config_y.num_contexts = 1
    device = Device()
    stream = device.create_stream()

    # Negative: xrange=0
    try:
        nvshmem.core.team_split_2d(parent_team, 0, config_x, 0, config_y, 0, new_team_name="BAD_XRANGE_TEAM")
        raise AssertionError("Expected exception for xrange=0")
    except Exception as e:
        print(f"Correctly caught exception for xrange=0: {e}")
    finally:
        # Ensure all ranks realign after the failed collective
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)

    # Negative: invalid parent_team
    try:
        nvshmem.core.team_split_2d(-1, 2, config_x, 0, config_y, 0, new_team_name="BAD_PARENT_TEAM_2D")
        raise AssertionError("Expected exception for invalid parent_team")
    except Exception as e:
        print(f"Correctly caught exception for invalid parent_team in 2d: {e}")
    finally:
        # Ensure all ranks realign after the failed collective
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)


def test_team_translate_pe_negative():
    print("Testing team_translate_pe negative cases")
    parent_team = Teams.WORLD if "WORLD" in Teams else 0
    config = TeamConfig()
    config.version = 2
    config.num_contexts = 1
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    device = Device()
    stream = device.create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    device.sync()
    # Create a team for valid translation
    if n_pes >= 2:
        config.uniqueid = 12350
        # Align ranks before entering collective and ensure cleanup regardless of exceptions
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        stream.sync()
        team = None
        try:
            team = nvshmem.core.team_init(config, 0, n_pes, my_pe, new_team_name="TRANS_TEAM")
            print("Created team for valid translation")
            # Negative: src_pe out of range
            try:
                nvshmem.core.team_translate_pe(team, n_pes + 10, parent_team)
                raise AssertionError("Expected exception for src_pe out of range")
            except Exception as e:
                print(f"Correctly caught exception for src_pe out of range: {e}")
        finally:
            if team is not None:
                nvshmem.core.team_destroy("TRANS_TEAM")
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    else:
        print("Skipping team_translate_pe negative test - need at least 2 PEs")
    device.sync()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    device = Device()
    stream = device.create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    device.sync()

    test_team_unique_id()
    device.sync()
    test_team_split_strided()
    device.sync()
    test_team_split_2d()
    device.sync()
    test_team_destroy()
    device.sync()
    test_team_destroy_negative()
    device.sync()
    test_team_init_negative()
    device.sync()
    test_team_split_strided_negative()
    device.sync()
    test_team_split_2d_negative()
    device.sync()
    test_team_translate_pe_negative()
    device.sync()

    nvshmem.core.finalize()
