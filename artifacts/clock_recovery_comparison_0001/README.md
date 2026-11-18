# Recorded clock-policy and MPC recovery comparison

[PNG](comparison.png) and [PDF](comparison.pdf) compare the existing 2 ms force audits. Both trajectories share exactly the first nine controls and 450 physics samples. The policy alone drops after 19 controls without a full turn; the fixed handoff to MPC completes a qualified turn after five corrections. The assisted result stops at the first successful control boundary, so the figure makes no claim about a later hold.

This plot adds no physics or learning. The policy is diagnostic clock BC with zero SAC updates, and the recovery is MPC-assisted. Input copies, source identities and the script are preserved here. Reproduce with:

```bash
uv run --frozen --with 'matplotlib==3.10.8' python artifacts/clock_recovery_comparison_0001/plot.py
```

The plotting dependency is isolated by `uv --with`; project dependencies and numerical sources are unchanged.
