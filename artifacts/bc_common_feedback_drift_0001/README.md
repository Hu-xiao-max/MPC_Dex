# Measured feedback drift at the existing noise floor

This is read-only analysis of the five archived stochastic trials at latent Gaussian standard deviation1e−5. It compares each actual sampled action with the policy's conditional mean at its visited state, and that conditional mean with the successful deterministic policy command at the same control index. Both differences use the same normalized action units. No training or physics was run.

At control1, all conditional means and observations match the deterministic trajectory exactly. At control2, the conditional-mean shift has L2 magnitude13.56–255.93times the direct sampled-action perturbation. Maximum mean-action shifts across the five seeds are .004213, .002959, .002070, .001482 and .0001933. The largest observation-coordinate changes are joint-velocity entries, between8.82e−5 and2.01e−4rad/s. This describes actual drift, not a mixed-unit Jacobian norm.

The separate fixed-nominal-mean noise experiment is needed to distinguish direct physical sensitivity from amplification through policy feedback. These ratios alone do not establish which mechanism causes failure or whether adding a time observation would resolve it.

`report.json` contains every available comparable control, per-group observation differences, source hashes and units. `analyze.py` reproduces the calculation from the six saved transition archives; `manifest.json` binds the artifact.
