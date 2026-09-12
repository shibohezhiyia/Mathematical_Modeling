# Synthetic calibration

`core.synthetic_calibration.assess_synthetic_interval_calibration` runs a
bounded interval-coverage protocol against a known synthetic generator. The
builder receives a seeded RNG, nominal coverage and replicate index, then
returns actual values and lower/upper bounds. Results include empirical
coverage, gap, finite-sample standard error and failed replicate count.

This is evidence that an implementation behaves as expected under the chosen
generator. It is not a posterior, a causal claim, or a distribution-free
coverage guarantee for real competition data. The generator and seed must be
stored with the experiment configuration.
