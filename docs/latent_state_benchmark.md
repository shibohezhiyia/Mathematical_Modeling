# Latent-state control benchmark

`assess_latent_false_positive_control` evaluates delay/PCA latent candidates on
a supplied high-noise independent generator. It reports the fraction of runs
where compression would be flagged under a declared threshold. This is an
operating-characteristic control for a heuristic, not a proof of a hidden
physical state; real latent claims still require an observation model or
independent measurements.
