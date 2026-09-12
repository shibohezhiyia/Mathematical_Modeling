# Cache performance benchmark

`run_cache_performance_benchmark` records cold, warm-first, warm-second and
incremental executions for the same input. It stores output digests, finite
cache-vs-uncached consistency checks, elapsed time and peak Python allocation.
It intentionally does not infer a speedup from one run; fixed cases, repeated
seeds and broader paired experiments are still required.
