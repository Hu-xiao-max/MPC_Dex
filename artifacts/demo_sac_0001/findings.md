# Demonstration-initialized SAC from cold reset: no cold success

**A fresh actor trained by BC on 12 verified cold-reset MPC demonstrations (221 rows, 10–32 controls each) drops the object at the second control in every cold trial, and 3745 online SAC updates from cold reset with 30/70 demonstration/online batches produce zero cold successes** in 1468 changing-policy training episodes and at every checkpoint. Stage status `complete` (exit 0, 503 s); final counters {'bc_updates': 12000, 'pretrain_updates': 2000, 'online_updates': 3745}.

## Actual execution

- Fresh `SACAgent` with the SAC0001 learner configuration; 12000 mean-only BC updates with episode-weighted sampling (BC loss 1.934 → 0.0280 in atanh space: the mean regression does not fit the mutually different demonstrations), then `initialize_std`.
- 2000 archived Soft-SARSA critic updates on demonstration rows; 4000 cold-reset interactions with one archived update each after 256 rows; sampled rewards divided by 200.
- Demonstrations: `/home/autel001/code/mpc_dex/artifacts/mpc_cold_demonstrations_0001` CEM seeds 100, 101, 102, 104, 105, 106, 110, 112, 120, 130, 131, 132, each stored only after a bit-exact independent physical replay with a separate force audit.

## Cold evaluations (deterministic + 10 stochastic, fresh environments, training RNG restored)

| Checkpoint | Deterministic | Stochastic | Controls before failure | Peak force-qualified angle (rad) |
| --- | --- | --- | --- | --- |
| bc_complete | fail (2 controls, drop) | 0/10 | 2–3 | 0.66 |
| pretrained | fail (2 controls, drop) | 0/10 | 2–3 | 0.66 |
| online_500 | fail (7 controls, drop) | 0/10 | 3–6 | 4.58 |
| online_1000 | fail (5 controls, drop) | 0/10 | 5–9 | 3.64 |
| online_1500 | fail (11 controls, drop) | 0/10 | 2–10 | 1.76 |
| online_2000 | fail (4 controls, drop) | 0/10 | 2–17 | 1.32 |
| online_2500 | fail (2 controls, palm_support) | 0/10 | 2–9 | 1.61 |
| online_3000 | fail (4 controls, drop) | 0/10 | 2–4 | 1.72 |
| online_3500 | fail (1 controls, drop) | 0/10 | 1–2 | 0.00 |
| online_4000 | fail (3 controls, drop) | 0/10 | 2–4 | 0.48 |

## Interpretation and scope

The demonstrations solve the turn with different regrasp sequences, so their mean action at the first controls is not a valid control: the classic multimodality failure of mean-only BC, not evidence against the demonstrations themselves (each completes the turn when replayed). The diagnostic prior is not the paper's SAC-derived primitive, and the online phase had no MPC scaffolding chunks; both are disclosed deviations. Nothing here is primitive-eligible. The demonstrations are reused as physically replayed prefixes in the multi-source suffix curriculum (`bootstrap_variant_0068`/`0069`).
