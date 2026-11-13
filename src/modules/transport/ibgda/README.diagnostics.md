# Experimental owned-QP diagnostics

`NVSHMEM_IBGDA_QP_DIAGNOSTICS=1` enables JSON records on standard error,
prefixed with `NVSHMEM_IBGDA_DIAG`. Diagnostics are disabled by default.
The prototype records the PE, process, transport generation, host, GPU UUID
and PCI address, NIC and port, QP type, local and remote QPN, RC peer, send
CQ, and send-queue capacity. Shared DCIs have no fixed peer attribution;
their records explicitly use `AMBIGUOUS_SHARED_DCI`. DCT records describe
an inbound endpoint, without assigning it a single sender.

Hardware counters are best effort and require the current DEVX context to
advertise queue-counter capacity, allocate a nonzero counter ID, accept
the association during RC RESET-to-INIT, and read that ID back from the
owned QP. The counter belongs to one process and NIC context and can be
shared by that process's RC QPs on the device. It is an aggregate, rather
than a per-QP counter. DCI and DCT counter association is outside this
prototype's scope. Records include accepted and verified attachment counts
so incomplete readback remains visible.

An unsupported capability, rejected allocation, zero ID, failed association
or failed query does not become a counter value of zero. Such records use
an explicit status and `counter_values: null`. Optional out-of-sequence
and retransmission fields require their own capability bits. Available
values are absolute 32-bit hardware counters; samples neither clear them
nor aggregate another counter set, and never address another vport. No
automatic wrap correction or per-peer attribution is inferred.

If counter association fails, baseline initialization is retried only when
querying that owned QP confirms it is still in RESET. The counter remains
owned until every accepted QP attachment has been successfully destroyed,
including attachments whose readback failed. Counter destruction is
attempted once; failure or an unresolved QP lifetime is reported explicitly.

`NVSHMEM_IBGDA_QP_COUNTER_INTERVAL_MS` defaults to 1000. Positive values
rate-limit queries from the existing CPU progress path. Zero disables
periodic sampling; initialization and finalization snapshots remain.
GPU-handler progress has no equivalent periodic host callback, so its
coverage is initialization and finalization only. Queries and logging are
outside the per-WQE device path.

Normal finalization also captures the software queue's reserved, ready,
producer and polled-consumer indices, before the device QP arrays are
freed. These are 64-bit **WQEBB** indices. One work request can use multiple
WQEBBs; these indices are not hardware error counters, hardware completion
timestamps, or a count of work requests. The published software producer
is read through the QP's actual producer pointer. It does not establish
when the NIC received a doorbell. Applications must follow NVSHMEM's normal
completion and finalization contract.

Build and run the CPU provider-lifecycle checks with:

```sh
cmake --build build --target host_ibgda_qcounter_lifecycle
./build/test/common/host_ibgda_qcounter_lifecycle
```

The checks also compile directly with a C++17 compiler. They use a fake
provider to verify command layouts, capability masks, rate limiting,
ownership, shared attachment lifetimes and failure handling. Their PASS
does not qualify actual hardware allocation, association, cleanup or
sampling overhead. Native communication, ownership readback and fault-path
testing on a counter-capable system are separate validation requirements.
