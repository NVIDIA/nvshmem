Performance tests accept command line arguments. Below is the full list of options, 
any given test will use only a subset of these options.
* -b, --min_size <minbytes> 
* -e, --max_size <maxbytes> 
* -f, --step <step factor for message sizes> 
* -n, --iters <number of iterations> 
* -w, --warmup_iters <number of warmup iterations> 
* -c, --ctas <number of CTAs to launch> (used in some device pt-to-pt tests) 
* -t, --threads_per_cta <number of threads per block> (used in some device pt-to-pt tests) 
* -d, --datatype: <int, int32_t, uint32_t, int64_t, uint64_t, long, longlong, ulonglong, size, ptrdiff, float, double, fp16, bf16> 
* -o, --reduce_op <min, max, sum, prod, and, or, xor> 
* -s, --scope <thread, warp, block, all> 
* -i, --stride stride between elements 
* -a, --atomic_op <inc, add, and, or, xor, set, swap, fetch_<inc, add, and, or, xor>, compare_swap> 
* --bidir: run bidirectional test 
* --msgrate: report message rate (MMPs)
* --dir: <read, write> (whether to run put or get operations) 
* --issue: <on_stream, host> (applicable in some host pt-to-pt tests) 
