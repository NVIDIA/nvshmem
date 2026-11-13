nvshmem_add_perftest(ep_rma_bench.cu)

# Unlike the pt-to-pt perftests, this benchmark drives MPI directly (init, rank,
# barrier) rather than going through nvshmem_perftest_helper's dlopen'd MPI
# function table, so it needs an explicit link. Whether it should instead be
# converted to the helper convention, or live under examples/, is an open
# question for review.
target_link_libraries(perf_device_ep_rma_bench MPI::MPI_C)
