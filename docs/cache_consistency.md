# Cache consistency

`compare_cached_uncached` compares two trusted execution routes while ignoring
telemetry fields such as timing and cache-hit markers. It reports numerical
differences, missing keys, and status changes as counterexamples. A finite pass
is only `tested_not_falsified`, never a global equivalence proof.
