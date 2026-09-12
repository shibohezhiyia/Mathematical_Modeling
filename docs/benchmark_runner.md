# Fixed benchmark runner

`run_fixed_benchmark` executes a bounded, predeclared case list and preserves
completed, failed, timed-out, and not-run-budget rows. It reports P50/P95/max
latency and status counts, but deliberately does not claim accuracy or success
rate without independent truth labels.
