# Saved suffix SAC outcomes

`plot.py` reads only the completed `bootstrap_suffix_sac_0001` generation. It checks the final manifest against every report used, copies those reports into `inputs`, and writes the exact plotted counts and source hashes to `plotted_data.json`. No training environment, learner or simulator is imported. No additional inference or physical episode is performed.

`outcomes.png` and `outcomes.pdf` separate:

- Completed online training episodes, with changing policies, in nonoverlapping groups of 100. A final partial group has its actual denominator and is marked. Frozen-actor warm-up episodes and any unfinished final suffix are excluded.
- Prefix-7 stochastic evaluations of the scheduled frozen checkpoints, using the same five prescribed seeds.
- Prefix-8 retention evaluations at baseline and final, including the separately reported deterministic outcome.
- Cold-reset deterministic evaluations and the final five stochastic cold attempts. These are policy-only and remain separate from the assisted suffix results.

All plotted success flags are checked for consistency with the saved strict force/no-support/no-drop summary fields. This does not replace the independent audit of the source physics. Ratios summarize these recorded trials and are not robustness estimates. Training success can span several changing learner versions and is not a frozen-policy success claim.

Render after source completion:

```bash
python artifacts/bootstrap_suffix_sac_plot_0001/plot.py
```

Rendering uses the existing system Python with Matplotlib 3.10.6. It does not install packages into or alter the numerical training environment. The generated manifest binds the plot source, copied report evidence, numeric plot data, PNG and PDF.
