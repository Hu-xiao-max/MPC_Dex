"""Stamp one frozen variant stage from this template; never execute physics or learning.

Usage: prepare.py --variant NAME   (or --list). Each variant becomes
artifacts/bootstrap_variant_<index>_<name>/ with its own hash-bound inputs.
Template files are pinned below; refresh the pins deliberately when the runner changes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np

T = Path(__file__).resolve().parent
A = T.parent
P = A / 'bootstrap_suffix_sac_0004'
EXCLUDED = ('execution_manifest.json', 'run.log')
PINS = {'run.py': 'a9c0ef8ae5737d4ebfeed0bd97889447904bebe7fcd388923d96eb466ea4839a', 'test_protocol.py': '76a8f5996e1fee46c0904fa9f40be984708b785eb571b1b8672557c6fd980a3c'}  # pinned by pin_template.py

BASE = dict(prefix_controls=6, prefix_set=[6], demonstration_include_mpc=True, demonstration_recorded_rows=[6, 9],
            pretrain_offline_fraction=.3, online_offline_fraction=.3, actor_bc_lambda=0., actor_bc_rows='all',
            actor_on_demonstration_states=True, handoff_noise_sigma=0., handoff_noise_prefixes=[], seed_matched=True,
            online_policy_interactions=2000, checkpoints=[500, 1000, 2000], initial_from=None,
            actor_bc_first_rows=1, actor_lr_override=None, eval_stochastic_trials=5, alpha_freeze=False, alpha_override=None,
            actor_bc_std_target=None, actor_bc_std_lambda=None, prefix_weights=None,
            prefix_sources=None, suffix_set=None, demonstration_suffix_rows=None, eval_source_count=3, handoff_noise_suffixes=[], holdout_source_count=0, suffix_weights=None,
            extra_sources=[], holdout_source_names=None, eval_source_names=None,
            start_prefix_set=None, start_prefix_weights=None, start_prefix_share=0., actor_bc_last_rows=None, actor_bc_shared_prefix_rows=True,
            spine_source=None, actor_bc_shared_lambda=None, online_demo_sarsa=False, labelled_handoff_share=0.,
            env_overrides=None, eval_recorded_prefixes=True, eval_cold_every_point=False, max_source_controls=None)
REC = dict(demonstration_include_mpc=False)  # recorded controls 7-9 only: the label that chains into the mastered prefix-7 state
VARIANTS = {
    'mixed_prefix': dict(index=1, prefix_set=[6, 7, 8],
        hypothesis='Forgetting is critic-led when the stream holds only prefix-6 failures; sampling the collection prefix uniformly from {6,7,8} keeps mastered states in replay. Keeps the 8-row MPC+recorded buffer for attribution against the recorded-only variants.'),
    'recorded_only_control': dict(index=2, **REC,
        hypothesis='Single-factor control: SAC0004 with the five MPC rows removed. Read-only probes show the MPC continuation is unfinishable by this actor from any point, so removing it should stop the drift toward the MPC region; retention loss is still expected (stream holds only prefix-6 failures).'),
    'recorded_bc10': dict(index=3, **REC, actor_bc_lambda=10.,
        hypothesis='Supervised pull (lambda 10, all three recorded rows) toward the unique recorded control 7, whose successor state the actor already solves 6/6; prefix-6 deterministic pass expected, stochastic limited by the narrow basin.'),
    'recorded_bc10_first': dict(index=4, **REC, actor_bc_lambda=10., actor_bc_rows='first',
        hypothesis='As recorded_bc10 but the BC term covers only the handoff row (obs6 -> recorded control 7), leaving the learned prefix-7/8 completion untouched by supervision.'),
    'recorded_bc10_mixed': dict(index=5, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10.,
        hypothesis='Supervised pull at the handoff plus mastered prefix-7/8 episodes in the stream against forgetting; the panel\'s highest-rated combination without new mechanics.'),
    'recorded_bc10_isolated_s6': dict(index=6, **REC, prefix_set=[7, 8], actor_bc_lambda=10., actor_on_demonstration_states=False,
        hypothesis='The handoff becomes a BC-only state: no on-policy episode starts at prefix 6 and demonstration rows are excluded from the SAC actor/temperature losses, so only the supervised term shapes the mean at obs6 while SAC on prefix-7/8 rows maintains the suffix; prefix 6 is still evaluated.'),
    'handoff_noise_mixed': dict(index=7, **REC, prefix_set=[6, 7, 8], handoff_noise_sigma=.025, handoff_noise_prefixes=[7, 8],
        hypothesis='Mastered competence is a point (basin about L2 .02-.05 around recorded control 7); executing the last prefix control with N(0,.025^2) action noise on prefixes 7/8 trains the suffix on a neighbourhood so that the stochastic policy\'s own deviation (about .27) stops capping prefix-6 success.'),
    'dart_recipe': dict(index=8, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., handoff_noise_sigma=.05, handoff_noise_prefixes=[7, 8], online_offline_fraction=.05,
        hypothesis='Combination judged jointly necessary by the panel: BC to recorded control 7 fixes the mean at obs6, sigma .05 handoff noise makes the continuation robust to policy-scale deviations, mixed prefixes preserve the suffix, and a 5% online demonstration share limits repetition; pretraining share stays .3.'),
    'dart_recipe_r2': dict(index=9, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., handoff_noise_sigma=.05, handoff_noise_prefixes=[7, 8], online_offline_fraction=.05, seed_matched=False,
        hypothesis='Seed replicate of dart_recipe (all seeds new) to separate design effect from seed luck at a knife-edge handoff.'),
    'recorded_bc10_mixed_r2': dict(index=10, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., seed_matched=False,
        hypothesis='Seed replicate of recorded_bc10_mixed.'),
    'low_fraction': dict(index=11, online_offline_fraction=.05,
        hypothesis='Only the online demonstration share drops from .3 to .05 (pretraining share unchanged), testing whether repeating 8 rows about 9.6 times per online batch destabilizes the actor.'),
    'no_actor_on_demo': dict(index=12, actor_on_demonstration_states=False,
        hypothesis='SAC actor and temperature losses restricted to on-policy rows; the critic still learns from demonstration rows. Tests whether actor updates on demonstration states drive the drift.'),
    'control_reseed': dict(index=13, seed_matched=False,
        hypothesis='SAC0004 settings with new seeds: measures seed variance of the SAC0004 failure and of retention loss at online 500.'),
    'full_curriculum_bc10': dict(index=14, **REC, prefix_set=[0, 1, 2, 3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[0, 9], actor_bc_lambda=10.,
        hypothesis='Reverse curriculum over every handoff state of the recorded trajectory at once, with the whole recorded trajectory as demonstrations and a supervised pull; the only variant that collects cold-start episodes and could show cold progress.'),
    # ---- wave 2 (2026-09-25): combine the wave-1 winners, replicate, densify checkpoints, extend budgets ----
    'bc_first_noise_mixed': dict(index=15, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., actor_bc_rows='first', handoff_noise_sigma=.025, handoff_noise_prefixes=[7, 8], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Wave-1 winners combined: first-row BC (recorded_bc10_first gave prefix-6 deterministic passes with full 7/8 retention) plus sigma .025 handoff noise with mixed prefixes (handoff_noise_mixed reached prefix-6 stochastic 5/5 at online 1000). Checkpoints every 250 to resolve the drift seen after 1000.'),
    'bc_first_noise_mixed_r2': dict(index=16, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., actor_bc_rows='first', handoff_noise_sigma=.025, handoff_noise_prefixes=[7, 8], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000], seed_matched=False,
        hypothesis='Seed replicate of bc_first_noise_mixed.'),
    'recorded_bc10_first_r2': dict(index=17, **REC, actor_bc_lambda=10., actor_bc_rows='first', checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000], seed_matched=False,
        hypothesis='Seed replicate of recorded_bc10_first with checkpoints every 250.'),
    'handoff_noise_mixed_r2': dict(index=18, **REC, prefix_set=[6, 7, 8], handoff_noise_sigma=.025, handoff_noise_prefixes=[7, 8], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000], seed_matched=False,
        hypothesis='Seed replicate of handoff_noise_mixed with checkpoints every 250 to test whether its online-1000 peak (prefix-6 5/5) recurs.'),
    'bc_first_mixed': dict(index=19, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., actor_bc_rows='first', checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='First-row BC with mixed prefixes and no handoff noise: isolates the contribution of noise in bc_first_noise_mixed.'),
    'bc_first_long': dict(index=20, **REC, actor_bc_lambda=10., actor_bc_rows='first', online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        hypothesis='recorded_bc10_first with twice the online budget: does the deterministic prefix-6 pass and the stochastic 3/5 stabilize or drift with more updates?'),
    'bc_first_noise_mixed_long': dict(index=21, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., actor_bc_rows='first', handoff_noise_sigma=.025, handoff_noise_prefixes=[7, 8], online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        hypothesis='bc_first_noise_mixed with twice the online budget.'),
    'bc_first_lambda3': dict(index=22, **REC, actor_bc_lambda=3., actor_bc_rows='first', checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Weaker first-row supervision (lambda 3) to test whether lambda 10 over-constrains the mean at the handoff.'),
    'bc_first_noise015_mixed': dict(index=23, **REC, prefix_set=[6, 7, 8], actor_bc_lambda=10., actor_bc_rows='first', handoff_noise_sigma=.015, handoff_noise_prefixes=[7, 8], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Narrower handoff noise (sigma .015) with first-row BC and mixed prefixes: the basin around recorded control 7 is about L2 .02-.05 wide, so .015 per dimension (L2 about .06) may widen it without swamping the suffix.'),
    # ---- wave 3: chained reverse curriculum, one recorded control earlier per stage ----
    'chain_p5_from_bc_first': dict(index=24, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first',
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained stage: inherit the recorded_bc10_first final checkpoint (prefix-6 deterministic pass, stochastic 3/5, prefix-7/8 retention 5/5), move the handoff one recorded control earlier with first-row BC toward recorded control 6 at obs5, and keep prefixes 6-8 in the stream against forgetting.'),
    'chain_p5_from_bc_first_noise': dict(index=25, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first', handoff_noise_sigma=.025, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As chain_p5_from_bc_first plus sigma .025 handoff noise on prefixes 6-8 to widen the basins the earlier stage established.'),
    # ---- wave 3a: stabilization and chained labels (independent of checkpoint selection) ----
    'bc_first_lowlr': dict(index=26, **REC, actor_bc_lambda=10., actor_bc_rows='first', actor_lr_override=1e-4, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='recorded_bc10_first with a three times smaller actor step: prefix-6 competence flickered between checkpoints in every BC-first run while training success kept rising, consistent with actor drift; ten stochastic trials per prefix reduce evaluation noise.'),
    'bc_first_lowlr_long': dict(index=27, **REC, actor_bc_lambda=10., actor_bc_rows='first', actor_lr_override=1e-4, eval_stochastic_trials=10, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        hypothesis='bc_first_lowlr with twice the budget, since a smaller step may need more updates to reach the same handoff residual.'),
    'chain_p5_first2': dict(index=28, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first_k', actor_bc_first_rows=2, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained prefix-5 stage keeping supervised labels at both new handoffs (obs5 -> recorded control 6 and obs6 -> recorded control 7): chain_p5_from_bc_first lost prefix-6 competence within 500 updates when only obs5 was supervised.'),
    'chain_p5_first2_noise': dict(index=29, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first_k', actor_bc_first_rows=2, eval_stochastic_trials=10, handoff_noise_sigma=.025, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='chain_p5_first2 plus sigma .025 handoff noise on prefixes 6-8.'),
    'chain_p5_first2_lowlr': dict(index=30, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first_k', actor_bc_first_rows=2, eval_stochastic_trials=10, actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='chain_p5_first2 with the smaller actor step.'),
    # ---- wave 3b: temperature control against the monotone alpha rise that inflates the handoff spread ----
    'bc_first_alphafreeze': dict(index=31, **REC, actor_bc_lambda=10., actor_bc_rows='first', alpha_freeze=True, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='recorded_bc10_first with the temperature frozen at its inherited value (.000306 normalized): alpha rose monotonically in every run because the policy entropy never approaches the -16 target, inflating the handoff std from .067 to .13-.17 while the mean stayed on the label; 30-trial rates drifted 20/30 -> 6/30 with more updates.'),
    'bc_first_alpha_low': dict(index=32, **REC, actor_bc_lambda=10., actor_bc_rows='first', alpha_freeze=True, alpha_override=1e-4, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Temperature frozen at a third of the inherited value (raw-equivalent .02), weakening the entropy bonus so SAC can shrink the std where failures occur.'),
    'bc_first_alphafreeze_lowlr': dict(index=33, **REC, actor_bc_lambda=10., actor_bc_rows='first', alpha_freeze=True, actor_lr_override=1e-4, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Both stabilizers: frozen temperature and the smaller actor step.'),
    'chain_p5_first2_alphafreeze': dict(index=34, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first_k', actor_bc_first_rows=2, alpha_freeze=True, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained prefix-5 stage with supervised labels at obs5 and obs6 and the temperature frozen at the inherited checkpoint value.'),
    'chain_p5_first2_alphafreeze_lowlr': dict(index=35, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=10., actor_bc_rows='first_k', actor_bc_first_rows=2, alpha_freeze=True, actor_lr_override=1e-4, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained prefix-5 stage with both stabilizers.'),
    # ---- wave 3c: pin the handoff harder (mean and spread) ----
    'bc_first_l100': dict(index=36, **REC, actor_bc_lambda=100., actor_bc_rows='first', eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Ten times stronger first-row supervision: 30-trial collapses coincide with the handoff mean drifting to L2 .09 from the label (0019 online_2000) in a basin about .025-.05 wide.'),
    'bc_first_std02': dict(index=37, **REC, actor_bc_lambda=10., actor_bc_rows='first', actor_bc_std_target=.02, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Spread anchor at the handoff row (latent std toward .02 per dimension, L2 about .08): the handoff std grew from .067 to .13-.17 in every BC run while the basin is about .025-.05 wide, capping stochastic success near two thirds.'),
    'bc_first_std01': dict(index=38, **REC, actor_bc_lambda=10., actor_bc_rows='first', actor_bc_std_target=.01, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Tighter spread anchor (.01 per dimension, L2 about .04, inside the measured basin).'),
    'bc_first_l100_std01': dict(index=39, **REC, actor_bc_lambda=100., actor_bc_rows='first', actor_bc_std_target=.01, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Strong mean pin plus tight spread anchor at the handoff row.'),
    'bc_first_l100_std01_alphafreeze': dict(index=40, **REC, actor_bc_lambda=100., actor_bc_rows='first', actor_bc_std_target=.01, alpha_freeze=True, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Strong mean pin, tight spread anchor and frozen temperature.'),
    # ---- wave 3d: replicate the pinned-handoff recipe and chain it to prefix 5 ----
    'bc_first_l100_std01_r2': dict(index=41, **REC, actor_bc_lambda=100., actor_bc_rows='first', actor_bc_std_target=.01, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000], seed_matched=False,
        hypothesis='Seed replicate of bc_first_l100_std01 (prefix-6 8/10 at online 1,750 with prefix-7/8 10/10).'),
    'bc_first_l100_std01_alphafreeze_r2': dict(index=42, **REC, actor_bc_lambda=100., actor_bc_rows='first', actor_bc_std_target=.01, alpha_freeze=True, eval_stochastic_trials=10, checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000], seed_matched=False,
        hypothesis='Seed replicate of bc_first_l100_std01_alphafreeze.'),
    'chain_p5_first2_l100_std01': dict(index=43, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained prefix-5 stage with the pinned-handoff recipe applied at both new handoffs (obs5 and obs6): strong mean pin and .01 spread anchor.'),
    'chain_p5_first2_l100_std01_alphafreeze': dict(index=44, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, alpha_freeze=True, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0004_recorded_bc10_first', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='chain_p5_first2_l100_std01 with the temperature frozen.'),
    'chain_p5_first2_l100_std01_from39': dict(index=45, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10,
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Chained prefix-5 stage inheriting the pinned-handoff checkpoint (bc_first_l100_std01 online_2000: prefix-6 deterministic pass and 9/10 stochastic, prefix-7/8 10/10, handoff std .01) with the same recipe at obs5 and obs6.'),
    'bc_first_l100_std01_long': dict(index=46, **REC, actor_bc_lambda=100., actor_bc_rows='first', actor_bc_std_target=.01, eval_stochastic_trials=10, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        hypothesis='bc_first_l100_std01 with twice the budget: does the pinned handoff stay stable beyond 2,000 updates (bc_first_long decayed from 20/30 to 6/30 by 4,000)?'),
    # ---- wave 3e: chained prefix-5 with the pinned recipe plus handoff noise on the later prefixes ----
    'chain_p5_pinned_noise025_from39': dict(index=47, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.025, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Pinned recipe at obs5/obs6 from the confirmed checkpoint plus sigma .025 noise on the final control of prefixes 6-8: the pinned chained stages keep prefix 6 but stay at 2-4/10 at prefix 5, while the wave-2 chained stage with noise on 6-8 reached the best prefix-5 rate (11/30), consistent with control 6 landing near rather than on the recorded prefix-6 state.'),
    'chain_p5_pinned_noise015_from39': dict(index=48, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As chain_p5_pinned_noise025_from39 with sigma .015.'),
    # ---- wave 3f: consolidate prefix 5 and open prefix 4 ----
    'chain_p5_pinned_noise015_long': dict(index=49, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        hypothesis='chain_p5_pinned_noise015_from39 (prefix-5 deterministic passes at four checkpoints, stochastic up to 5/10, prefixes 6-8 retained) with twice the budget: the pinned prefix-6 recipe improved monotonically to 4,000 updates.'),
    'chain_p5_pinned_noise015_r2': dict(index=50, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8], seed_matched=False,
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Seed replicate of chain_p5_pinned_noise015_from39.'),
    'chain_p5_pinned_noise01_from39': dict(index=51, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.01, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Sigma .01 handoff noise: sigma .025 collapsed every prefix in the chained stage while .015 worked; .01 matches the pinned policy spread.'),
    'chain_p4_pinned_noise015_from48': dict(index=52, **REC, prefix_controls=4, prefix_set=[4, 5, 6, 7, 8], demonstration_recorded_rows=[4, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=3, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[5, 6, 7, 8],
        initial_from=['bootstrap_variant_0048_chain_p5_pinned_noise015_from39', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Next reverse-curriculum step: prefix 4 inherited from the prefix-5 stage (online 2,000: prefix-5 deterministic pass, 3/10 stochastic; prefix 6 9/10; 7-8 10/10), with recorded controls 5-7 pinned at obs4-obs6 and sigma .015 noise on prefixes 5-8.'),
    # ---- wave 3g: raise the prefix-5 rate (30-trial: 11-13/30 with prefixes 6-8 retained) ----
    'chain_p5_pinned_w55': dict(index=53, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], prefix_weights=[.55, .15, .15, .15], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='More on-policy prefix-5 episodes (55% of draws) so SAC sees more successes and failures at the knife-edge handoff.'),
    'chain_p5_pinned_l1000': dict(index=54, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=1000., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Ten times stronger pin at obs5/obs6: the recorded control 6 basin is narrower than .01 while the pinned mean sits .01-.03 away.'),
    'chain_p5_pinned_std005': dict(index=55, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.005, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8],
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Tighter spread anchor (.005) at the two pinned handoffs.'),
    'chain_p5_pinned_lowlr': dict(index=56, **REC, prefix_controls=5, prefix_set=[5, 6, 7, 8], demonstration_recorded_rows=[5, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=2, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0039_bc_first_l100_std01', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Smaller actor step to reduce jitter of the pinned mean around the knife-edge label.'),
    # ---- wave 3h: prefix 4 from the best prefix-5 stage ----
    'chain_p4_pinned_lowlr_from56': dict(index=57, **REC, prefix_controls=4, prefix_set=[4, 5, 6, 7, 8], demonstration_recorded_rows=[4, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=3, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0056_chain_p5_pinned_lowlr', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Prefix 4 inherited from the prefix-5 stage with the smaller actor step (online 2,000 in-run: prefix-5 deterministic pass and 7/10, prefix-6 10/10, 7-8 10/10), pinning recorded controls 5-7 at obs4-obs6, sigma .015 noise on prefixes 5-8, actor step 1e-4.'),
    'chain_p4_pinned_noise01_lowlr_from51': dict(index=58, **REC, prefix_controls=4, prefix_set=[4, 5, 6, 7, 8], demonstration_recorded_rows=[4, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=3, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.01, handoff_noise_prefixes=[5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0051_chain_p5_pinned_noise01_from39', 'online_1750'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Prefix 4 inherited from the best 30-trial prefix-5 checkpoint (0051 online_1750: prefix-5 22/30 D, prefix-6 25/30 D, prefix-7/8 30/30), sigma .01 noise on prefixes 5-8, recorded controls 5-7 pinned, smaller actor step against the disruption of later prefixes seen in 0052.'),
    # ---- wave 3i: prefix 3 ----
    'chain_p3_pinned_lowlr_from57': dict(index=59, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0057_chain_p4_pinned_lowlr_from56', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Prefix 3 inherited from the prefix-4 stage (0057 online_2000 in-run: prefix-4 deterministic pass and 3/10, prefix-5 8/10, prefixes 6-8 10/10), recorded controls 4-7 pinned at obs3-obs6, sigma .015 noise on prefixes 4-8, actor step 1e-4.'),
    'chain_p3_pinned_noise01_lowlr_from58': dict(index=60, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.01, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Prefix 3 inherited from the sigma .01 prefix-4 stage (0058 online_2000 in-run: prefix-4 1/10, prefix-5 deterministic pass and 5/10, prefixes 6-8 10/10).'),
    # ---- wave 3j: prefix 3 pushes and full-curriculum attempts with the pinned low-step recipe ----
    'chain_p3_cont_from59': dict(index=61, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0059_chain_p3_pinned_lowlr_from57', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Continue the prefix-3 stage for another 2,000 updates from its final checkpoint (prefix-3 0-2/10 in-run, later prefixes retained), since prefix 4 needed more than one stage-length to appear.'),
    'chain_p3_std005_l300_from59': dict(index=62, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=300., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.005, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0059_chain_p3_pinned_lowlr_from57', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Tighter pin (lambda 300, std .005) at obs3-obs6 continuing from the prefix-3 stage: deterministic prefix-3 trials pass at some checkpoints while stochastic ones fail, pointing at spread and mean precision at the knife-edge control 4.'),
    'full_pinned_lowlr_noise_from57': dict(index=63, **REC, prefix_controls=0, prefix_set=[0, 1, 2, 3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[0, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=7, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0057_chain_p4_pinned_lowlr_from56', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='All remaining handoffs at once from the prefix-4 stage: recorded controls 1-7 pinned at obs0-obs6 with the .01 spread anchor, uniform prefixes 0-8 in the stream (cold-start episodes included), sigma .015 noise on prefixes 4-8, actor step 1e-4; the only stage since 0014 that collects cold episodes.'),
    'full_pinned_lowlr_from57': dict(index=64, **REC, prefix_controls=0, prefix_set=[0, 1, 2, 3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[0, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=7, actor_bc_std_target=.01, eval_stochastic_trials=10, actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0057_chain_p4_pinned_lowlr_from56', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As full_pinned_lowlr_noise_from57 without handoff noise.'),
    'chain_p3_std002_l1000_from59': dict(index=65, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=1000., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.002, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0059_chain_p3_pinned_lowlr_from57', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Near-deterministic early handoffs: the basin probe with the 0059 policy shows L2 .005 deviations of recorded controls 4 and 5 already fail 5/8 and 2/8, so the spread anchor moves to .002 (L2 about .008) and the pin to lambda 1,000 at obs3-obs6.'),
    'chain_p3_std002_l1000_lr3e5_from59': dict(index=66, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=1000., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.002, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=3e-5,
        initial_from=['bootstrap_variant_0059_chain_p3_pinned_lowlr_from57', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As chain_p3_std002_l1000_from59 with a ten-times smaller actor step than the default, to hold the pinned means within the sub-.005 basins.'),
    'chain_p3_cont2_from61': dict(index=67, **REC, prefix_controls=3, prefix_set=[3, 4, 5, 6, 7, 8], demonstration_recorded_rows=[3, 9], actor_bc_lambda=100., actor_bc_rows='first_k', actor_bc_first_rows=4, actor_bc_std_target=.01, eval_stochastic_trials=10, handoff_noise_sigma=.015, handoff_noise_prefixes=[4, 5, 6, 7, 8], actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0061_chain_p3_cont_from59', 'online_1750'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Third stage-length at prefix 3 from the best 30-trial prefix-3 checkpoint so far (0061 online_1750: prefix-3 8/30 D, prefix-4 16/30 D, prefix-5 16/30, prefix-6 28/30, prefixes 7-8 30/30): prefix 3 rose from 3/30 to 8/30 over one continuation.'),
    # ---- wave 4: multi-source suffix curriculum over verified cold-reset MPC demonstrations ----
    'ms_suffix123_from61': dict(index=68, **REC, prefix_controls=6, prefix_set=[6, 7, 8], prefix_sources=['mpc_cold_demonstrations_0001', 15], suffix_set=[1, 2, 3], demonstration_suffix_rows=3, eval_source_count=3,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=5, actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0061_chain_p3_cont_from59', 'online_1750'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Diverse handoff states: each episode replays a random verified cold-reset MPC demonstration up to its last 1-3 controls and the policy finishes; the last three rows of every demonstration are pinned (mean and .01 spread) and serve as critic demonstrations. Inherits the chained checkpoint that already solves recorded prefixes 4-8; recorded prefixes 6-8 are evaluated for retention.'),
    'ms_suffix123_noise_from61': dict(index=69, **REC, prefix_controls=6, prefix_set=[6, 7, 8], prefix_sources=['mpc_cold_demonstrations_0001', 15], suffix_set=[1, 2, 3], demonstration_suffix_rows=3, eval_source_count=3,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=5, actor_lr_override=1e-4, handoff_noise_sigma=.015, handoff_noise_suffixes=[2, 3],
        initial_from=['bootstrap_variant_0061_chain_p3_cont_from59', 'online_1750'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As ms_suffix123_from61 with sigma .015 noise on the handoff control for suffixes 2 and 3.'),
    'ms_suffix1to6_from68': dict(index=70, **REC, prefix_controls=6, prefix_set=[6, 7, 8], prefix_sources=['mpc_cold_demonstrations_0001', 25], suffix_set=[1, 2, 3, 4, 5, 6], demonstration_suffix_rows=6, eval_source_count=3, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=5, actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0068_ms_suffix123_from61', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Second multi-source stage: suffix lengths 1-6 over 23 training demonstrations with the last six rows pinned, two demonstrations held out of training and evaluated only, inheriting the suffix-1-3 stage.'),
    'ms_suffix1to6_noise_from68': dict(index=71, **REC, prefix_controls=6, prefix_set=[6, 7, 8], prefix_sources=['mpc_cold_demonstrations_0001', 25], suffix_set=[1, 2, 3, 4, 5, 6], demonstration_suffix_rows=6, eval_source_count=3, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=5, actor_lr_override=1e-4, handoff_noise_sigma=.015, handoff_noise_suffixes=[3, 4, 5, 6],
        initial_from=['bootstrap_variant_0068_ms_suffix123_from61', 'online_2000'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='As ms_suffix1to6_from68 with sigma .015 handoff noise for suffixes 3-6.'),
    'ms_suffix4to10_from70': dict(index=72, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 60], suffix_set=[4, 5, 6, 7, 8, 9, 10], demonstration_suffix_rows=10, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4,
        initial_from=['bootstrap_variant_0070_ms_suffix1to6_from68', 'online_500'], checkpoints=[250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        hypothesis='Third multi-source stage: suffix lengths 4-10 (demonstrations of at most 10 controls become cold-start episodes), last ten rows pinned, every available demonstration except two held-out ones used for training (the suffix-1-3 checkpoint memorized per-trajectory suffixes: 61/120, 0/120, 3/120 on unseen demonstrations), lighter in-run evaluation (two training sources plus the two held-out, three stochastic trials) with the 30-trial evaluator for the rest. Inherits the suffix-1-6 stage at online 500 (its best in-run point) rather than waiting for its slow completion.'),
    'ms_suffix1to6_all_long_from68': dict(index=73, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 60], suffix_set=[1, 2, 3, 4, 5, 6], demonstration_suffix_rows=6, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0068_ms_suffix123_from61', 'online_2000'],
        hypothesis='Alternative second stage: suffix lengths 1-6 over every available demonstration (34 training, 2 held out) for 4,000 updates from the suffix-1-3 checkpoint, against 0070 (23 sources, 2,000 updates) whose suffix-3 rate fell while longer suffixes barely started.'),
    'ms_suffix1to6_cont_from70': dict(index=74, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 60], suffix_set=[1, 2, 3, 4, 5, 6], demonstration_suffix_rows=6, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0070_ms_suffix1to6_from68', 'online_2000'],
        hypothesis='Continue the suffix-1-6 stage for 4,000 more updates over every available demonstration: its rates were still rising at online 1,500 (20/25, 16/25, 10/25, 6/25, 6/25, 4/25).'),
    'ms_suffix4to10_cont_from72': dict(index=75, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 60], suffix_set=[4, 5, 6, 7, 8, 9, 10], demonstration_suffix_rows=10, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0072_ms_suffix4to10_from70', 'online_2000'],
        hypothesis='Continue the suffix-4-10 stage for 4,000 more updates.'),
    'ms_suffix1to8_from74': dict(index=76, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 80], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], demonstration_suffix_rows=8, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0074_ms_suffix1to6_cont_from70', 'online_4000'],
        hypothesis='Multi-source curriculum step requested by root: suffix lengths 1-8 over every available demonstration (two held out), last eight rows pinned, inheriting 0074 online_4000 (30-trial: 240/240, 168/240, 134/240, 49/240, 43/240, 23/240 for suffix lengths 1-6 on eight demonstrations).'),
    'ms_suffix1to8_cont_from76': dict(index=77, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 80], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], demonstration_suffix_rows=8, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0076_ms_suffix1to8_from74', 'online_4000'],
        hypothesis='Continue the suffix-1-8 stage for 4,000 more updates: 0076 regressed at 30 trials (180/240, 150/240, 58/240 for lengths 1-3 versus 240/168/134 before), as 0074 did before recovering within one stage-length.'),
    'ms_suffix1to7_from74': dict(index=78, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 80], suffix_set=[1, 2, 3, 4, 5, 6, 7], demonstration_suffix_rows=7, eval_source_count=2, holdout_source_count=2,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0074_ms_suffix1to6_cont_from70', 'online_4000'],
        hypothesis='Smaller curriculum step: suffix lengths 1-7 from 0074 online_4000 (one added control instead of two).'),
    'ms_suffix1to8_weighted_from74': dict(index=79, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8, eval_source_count=2, holdout_source_count=3,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0074_ms_suffix1to6_cont_from70', 'online_4000'],
        hypothesis='Root-requested rerun from 0074 online_4000 over the enlarged demonstration set with suffix sampling weights 2:2:2:2:1:1:1:1 for lengths 1-8, so the short suffixes keep the share they had in the 1-6 stage (uniform 1-8 traded short-suffix competence for little gain at longer suffixes in 0076/0077); three demonstrations held out.'),
    'ms_dagger_from79': dict(index=80, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], extra_sources=[['mpc_dagger_labels_0001', 200]],
        holdout_source_names=['seed_194', 'seed_195', 'seed_196', 'seed_197'], eval_source_names=['seed_100', 'seed_101'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0079_ms_suffix1to8_weighted_from74', 'online_4000'],
        hypothesis='DAgger-style aggregation: the 0079 policy fails on every held-out demonstration (seeds 194-196, 0/90 at every suffix length) because it emits controls that release the object at states it has never seen. The verified cold-reset MPC was run from the states the frozen 0079 policy actually visits on those demonstrations (mpc_dagger_labels_0001: demonstration prefix, 0-3 policy controls, then MPC to a strict force-qualified success, independently replayed bit-for-bit). Those episodes join the training sources; their demonstration rows are the MPC segment after the last policy control only. The three labelled demonstrations and the never-seen seed_197 are evaluated as held-out sources. Continues 0079 online_4000 with the same weighted suffix-1-8 recipe.'),
    'ms_add_heldout_demos_from79': dict(index=81, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120],
        holdout_source_names=['seed_197'], eval_source_names=['seed_100', 'seed_101', 'seed_194', 'seed_195', 'seed_196'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8,
        actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0079_ms_suffix1to8_weighted_from74', 'online_4000'],
        hypothesis='Control for the DAgger stage: the same continuation of 0079 online_4000 with the previously held-out demonstrations (seeds 194-196) simply added as ordinary training sources, without labels at policy-visited states; only seed_197 stays held out. Separates the effect of labelling policy-visited states from the effect of adding those three trajectories.'),
    'ms_shared_prefix_from81': dict(index=82, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120],
        holdout_source_names=['seed_197'], eval_source_names=['seed_100', 'seed_101'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40,
        start_prefix_set=[0, 1, 2, 3], start_prefix_weights=[1, 2, 1, 1], start_prefix_share=.5, actor_bc_rows='per_item', actor_bc_last_rows=8,
        actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0081_ms_add_heldout_demos_from79', 'online_4000'],
        hypothesis='Longer-suffix stage with shared cold-start prefixes. The demonstrations form a tree from the common cold reset: 34 of the 63 training sources share a bit-identical first control (35 of the 64 in the collection; the 35th is the held-out seed_197), 8 of those the first two and 4 the first three. Half of the episodes hand off at those shared states (k = 0 cold reset, 1, 2, 3 controls from the reset, weights 1:2:1:1) and the policy completes the whole remaining turn (8-32 controls), so one handoff state carries many verified continuations instead of one; the other half keep the weighted end-suffix episodes (lengths 1-8). Whole demonstrations enter the demonstration buffer (entropy-free critic targets along every continuation during critic pretraining; the online batches carry the same demonstration rows under the standard SAC backup, as in every stage of this family). The actor BC term covers the last 8 rows of each demonstration plus its shared prefix rows, whose labels agree across the group; a tail row that lies at a state another source also visits with a different control (the two 8-control and one 9-control demonstrations) is never a BC target, so no supervised state carries conflicting labels. Continues 0081 online_4000 (63 training sources incl. 194-196; seed_197 held out).'),
    'ms_shared_prefix_nosharedbc_from81': dict(index=83, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120],
        holdout_source_names=['seed_197'], eval_source_names=['seed_100', 'seed_101'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40,
        start_prefix_set=[0, 1, 2, 3], start_prefix_weights=[1, 2, 1, 1], start_prefix_share=.5, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_shared_prefix_rows=False,
        actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        initial_from=['bootstrap_variant_0081_ms_add_heldout_demos_from79', 'online_4000'],
        hypothesis='Ablation of the shared-prefix stage (index 82): identical episode stream and demonstration buffer, but the actor BC term excludes the shared prefix rows (BC on the last 8 rows only, with the same exclusion of tail rows at shared states), so the BC term never touches the cold reset or the shared states; any cold-start or shared-state gain in 82 that is absent here is attributable to supervision toward the shared first controls rather than to the long-suffix episodes and the whole-trajectory critic data.'),
    'ms_spine193_sarsa_from82': dict(index=84, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], holdout_source_names=['seed_197'], eval_source_names=['seed_100'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], initial_from=['bootstrap_variant_0082_ms_shared_prefix_from81', 'online_4000'], start_prefix_set=list(range(14)), start_prefix_share=.5, spine_source='seed_193', actor_bc_shared_lambda=50., online_demo_sarsa=True,
        hypothesis='Spine curriculum with two repairs of the mechanisms that stalled 0082. (1) Online, demonstration rows now bootstrap along their recorded next action without the entropy term (online_demo_sarsa; the pretraining target kept online), so the value of every demonstration path is its own return rather than the failing policy continuation: Q along the paths becomes a ridge the actor can climb instead of a surface that is low everywhere the policy fails. (2) The shared-prefix and spine states get their own BC term: per distinct state (14 here: the cold reset, the k = 1 and k = 2 states, and the 11 later spine states) lambda_shared 50 times the mean-squared error of the actor mean and of the log-std, summed over states, so each supervised state carries weight 50 whatever its multiplicity, in the 25-100 per-row range of the single-handoff chain stages; in 0082 the 547-row average gave the cold reset 34/547*100 = 6 and other shared states 0.2-0.7, and the actor sat 0.12-0.20 from the shared controls (basins .02-.05). (3) One demonstration, seed_193 (14 controls; shares 3 controls with 3 other sources, 1 with 33; the 0082 policy completes its last 4 controls deterministically and 3/3 stochastically), is the spine: every one of its rows is a supervised row and half of the episodes hand off at its prefixes k = 0..13 (uniform; k <= 3 are the shared states), so the chain back to the cold reset follows one continuation while the other half of the episodes keep the weighted end-suffix competence over all 63 sources. Evaluated at every spine prefix, the cold reset and the held-out seed_197.'),
    'ms_spine193_nosarsa_from82': dict(index=85, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], holdout_source_names=['seed_197'], eval_source_names=['seed_100'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], initial_from=['bootstrap_variant_0082_ms_shared_prefix_from81', 'online_4000'], start_prefix_set=list(range(14)), start_prefix_share=.5, spine_source='seed_193', actor_bc_shared_lambda=50.,
        hypothesis='Ablation of index 84 without the online demonstration backup (standard SAC targets on demonstration rows, as in every earlier stage): isolates the contribution of the value ridge along the demonstration paths from the spine curriculum and the stronger pins. Caveat: the backup differs from the standard one in two ways at once, the recorded successor instead of a policy sample and no entropy term (alpha * log pi is about 0.3-0.5 normalized units per step at the inherited temperature and spread), so this pair attributes their joint effect; stochastic in-run cells use index-derived seeds and only the 30-trial frozen evaluations on fixed seeds support stochastic comparisons.'),
    'ms_shared_sarsa_from82': dict(index=86, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], holdout_source_names=['seed_197'], eval_source_names=['seed_100'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], initial_from=['bootstrap_variant_0082_ms_shared_prefix_from81', 'online_4000'], start_prefix_set=[0, 1, 2, 3], start_prefix_weights=[1, 2, 1, 1], start_prefix_share=.5, actor_bc_shared_lambda=50., online_demo_sarsa=True,
        hypothesis='Ablation of index 84 without the spine (the 0082 shared-state stream, k = 0..3 over the shared groups) but with the online demonstration backup and the stronger shared-row pins: isolates the contribution of following one continuation from the two learning-signal repairs.'),
    'ms_spine193_tube_from84': dict(index=87, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], extra_sources=[['mpc_spine_tube_0001/tracking', 400]], holdout_source_names=['seed_197'], eval_source_names=['seed_100', 'track_seed_193_j4_r0300_d0', 'track_seed_193_j7_r0100_d1'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40, actor_bc_rows='per_item', actor_bc_last_rows=40, actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], initial_from=['bootstrap_variant_0084_ms_spine193_sarsa_from82', 'online_4000'], start_prefix_set=list(range(14)), start_prefix_share=.4, spine_source='seed_193', actor_bc_shared_lambda=50., online_demo_sarsa=True, labelled_handoff_share=.5,
        hypothesis='Feedback law around the spine from coherent MPC corrections. The demonstrations are chaotic open-loop sequences (spine probe: L2 .003 perturbations of controls 0-7 of seed_193 fail every open-loop replay; the 0084 policy, within 0.00-0.05 of every spine control, fails from prefixes 4-8), so the policy must correct, and the corrections it is taught must be local: mpc_spine_tube_0001/tracking perturbs one spine control by L2 .01/.03/.1 in four directions and lets a reference-tracking CEM (warm start = the spine\'s own upcoming controls, sampling floors x 0.1) complete the turn, giving corrections 0.2-0.6 from the spine control and from each other instead of the 2.8-3.5 of the re-planning CEM. Those verified branches join the sources: their MPC rows are BC targets (all rows of the correction segment; the perturbed control never), half of the episodes drawn from them hand off at the first corrected state (labelled_handoff_share .5), 40 % of all episodes hand off at the spine prefixes 0-13, the rest are weighted end-suffix episodes over the originals and the branches. Continues 0084 online_4000 (frontier at spine prefix 9). Evaluated at every spine prefix, the cold reset, seed_100, two branches and the held-out seed_197.'),
    'ms_spine193_tube_noperturbhandoff_from84': dict(index=88, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['mpc_cold_demonstrations_0001', 120], extra_sources=[['mpc_spine_tube_0001/tracking', 400]], holdout_source_names=['seed_197'], eval_source_names=['seed_100', 'track_seed_193_j4_r0300_d0', 'track_seed_193_j7_r0100_d1'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=40, actor_bc_rows='per_item', actor_bc_last_rows=40, actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], initial_from=['bootstrap_variant_0084_ms_spine193_sarsa_from82', 'online_4000'], start_prefix_set=list(range(14)), start_prefix_share=.4, spine_source='seed_193', actor_bc_shared_lambda=50., online_demo_sarsa=True, labelled_handoff_share=0.,
        hypothesis='Ablation of index 87 without the handoffs at the first corrected state: the branches enter only through the weighted end-suffix episodes and the BC rows. Attributes any gain of 87 at the spine prefixes to on-policy practice from the perturbed states versus the supervision alone.'),
    'cyl_ms_from82': dict(index=89, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_count=3, eval_source_count=2, suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0082_ms_shared_prefix_from81', 'online_4000'],
        hypothesis='Route (b), object geometry: the archived environment with object_shape circular (40 mm, the archived diameter; grasp, mass, friction, gains, period unchanged). The hexagon demonstrations are open-loop intolerant over five to nine controls per turn because its edges rolling over fingertips are bifurcation events; cylinder demonstrations carry three to six such decisions (object_shape_0001). The multi-source weighted suffix curriculum that reached 85/59/28 % for the last 1/2/3 controls on the hexagon (0079-0082) is run on verified cylinder demonstrations (archived collector contract), inheriting the best hexagon multi-source checkpoint (0082 online_4000) so that the end-of-turn competence transfers; three sources held out; cold reset evaluated at every checkpoint. Success criterion for the campaign: a frozen checkpoint completing the turn from the cold reset.'),
    'cyl_ms_from_sac0001': dict(index=90, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_count=3, eval_source_count=2, suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=None,
        hypothesis='Control for index 89: the same cylinder curriculum from the archived SAC0001 prefix-7 checkpoint (the default initial checkpoint of this family) instead of the hexagon multi-source checkpoint, to separate what the inherited hexagon competence contributes on the new object. Caveat: the two inherited checkpoints also differ in their learner counters and temperature (SAC0001 starts the actor-step ramp at 0.4 of 1e-4 and alpha at 3e-4 raw-equivalent, 0082 at the full step and 6e-3), so the pair attributes the joint effect of weights, ramp and temperature.'),
    'cyl_ms_s123_from82': dict(index=91, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 24], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_101', 'seed_102'],
        suffix_set=[1, 2, 3], demonstration_suffix_rows=3, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=2000, checkpoints=[500, 1000, 1500, 2000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0082_ms_shared_prefix_from81', 'online_4000'],
        hypothesis='Staged start of the cylinder curriculum: on the hexagon the multi-source route only progressed when the suffix set grew gradually (0068: lengths 1-3 over 15 sources reached 10/15, 10/15, 9/15 by 1,000 updates; the direct 1-8 stages regressed), whereas 0089 jumped to lengths 1-8 on the cylinder from the transferred checkpoint. Same 24-source snapshot and held-out sources as 0089/0090 (sorted names, first 24), lengths 1-3 uniform, three demonstration rows per source, 2,000 updates, inheriting the hexagon checkpoint 0082 online_4000; the follow-up extends to 1-6 and 1-8 from its best checkpoint as 0070/0074 did.'),
    'cyl_ms_s123_from89': dict(index=92, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 24], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_101', 'seed_102'],
        suffix_set=[1, 2, 3], demonstration_suffix_rows=3, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=2000, checkpoints=[500, 1000, 1500, 2000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0089_cyl_ms_from82', 'online_2500'],
        hypothesis='Consolidation candidate: the best cylinder checkpoint so far (0089 online_2500: 90/150 at the last control over five evaluated sources, 60/90 on the three held-out ones, 48/150 at the last two) continued with the narrow suffix set 1-3 on the same 24-source snapshot, against 0091 which starts the same staged curriculum from the hexagon checkpoint. The better of the two at 30 trials seeds the extension to lengths 1-6.'),
    'cyl_ms_s1to6_from89': dict(index=93, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 24], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_101', 'seed_102'],
        suffix_set=[1, 2, 3, 4, 5, 6], suffix_weights=[2, 2, 2, 1, 1, 1], demonstration_suffix_rows=6, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0089_cyl_ms_from82', 'online_2500'],
        hypothesis='Extension of the cylinder curriculum to suffix lengths 1-6 (weights 2:2:2:1:1:1, so the short suffixes keep two thirds of the episodes) from the reference cylinder checkpoint 0089 online_2500 (90/150, 48/150, 35/150, 15/150 at lengths 1-4; held-out 60/90 at the last control; cold 0/30), on the same 24-source snapshot, for 4,000 updates as the hexagon extension 0074 needed. The staged start from the hexagon checkpoint (0091) did not beat the direct jump on the cylinder; this stage tests whether lengths 4-6 become reliable with the extra updates, with the cold reset evaluated at every checkpoint.'),
    'cyl_ms_s1to6_all_from89': dict(index=94, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_101', 'seed_102'],
        suffix_set=[1, 2, 3, 4, 5, 6], suffix_weights=[2, 2, 2, 1, 1, 1], demonstration_suffix_rows=6, actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0089_cyl_ms_from82', 'online_2500'],
        hypothesis='Same as index 93 (lengths 1-6 weighted from 0089 online_2500, 4,000 updates) over every verified cylinder demonstration (54 at stamp time; 51 training) instead of the 24-source snapshot, with the same three held-out sources, to measure what source count contributes on the cylinder, as 0074 did on the hexagon.'),
    'cyl_ms_s1to8_cont_from94': dict(index=95, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0094_cyl_ms_s1to6_all_from89', 'online_4000'], eval_source_names=['seed_101', 'seed_172'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8], suffix_weights=[2, 2, 2, 2, 1, 1, 1, 1], demonstration_suffix_rows=8,
        hypothesis='Continuation of the all-source cylinder stage (0094 online_4000: 120/150 at the last control, held-out 60/90; 31, 27, 5 of 150 at lengths 2-4) with lengths 1-8 weighted 2:2:2:2:1:1:1:1 for 4,000 more updates, as the hexagon continuation 0074 gained a length; evaluated on a 10-control source (seed_101) and the shortest 8-control source (seed_172) plus the held-out trio.'),
    'cyl_ms_s1to9_long_from94': dict(index=96, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0094_cyl_ms_s1to6_all_from89', 'online_4000'], eval_source_names=['seed_172', 'seed_116', 'seed_174'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9], suffix_weights=[1, 1, 1, 2, 2, 2, 2, 2, 2], demonstration_suffix_rows=9,
        hypothesis='Long-suffix emphasis from 0094 online_4000: lengths 1-9 weighted 1:1:1:2:2:2:2:2:2 over all 54 sources, so two thirds of the episodes hand off at least four controls before the end and, for the 8-9-control sources (seeds 172, 116, 174), a suffix of 8-9 is the cold reset itself; evaluated on those three shortest sources plus the held-out trio, with the cold reset at every checkpoint. Tests whether the frontier can be pushed through the short sources, whose intolerant prefix is three to six decisions.'),
    'cyl_ms_s1to9_long_cont_from96': dict(index=97, **REC, prefix_controls=6, prefix_set=[8], actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0096_cyl_ms_s1to9_long_from94', 'online_4000'], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_172', 'seed_116', 'seed_174'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9], suffix_weights=[1, 1, 1, 2, 2, 2, 2, 2, 2], demonstration_suffix_rows=9,
        hypothesis='Continuation of 0096 (lengths 1-9 weighted toward long suffixes over all 54 sources) for 4,000 more updates from its online_4000, whose in-run cells reached deterministic passes at lengths 5-6 on the 8-control source seed_172 (its cold reset is length 8) and 1-3 on seed_116; same evaluated and held-out sources.'),
    'cyl_ms_short_s1to10_from96': dict(index=98, **REC, prefix_controls=6, prefix_set=[8], actor_bc_lambda=100., actor_bc_rows='all', actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0096_cyl_ms_s1to9_long_from94', 'online_4000'], prefix_sources=['cylinder_demonstrations_0001', 120], max_source_controls=12, holdout_source_names=['seed_155'], eval_source_names=['seed_172', 'seed_116', 'seed_174'], suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], suffix_weights=[1, 1, 1, 2, 2, 2, 3, 3, 3, 3], demonstration_suffix_rows=10,
        hypothesis='Concentration on the short cylinder demonstrations (at most 12 controls: 21 sources, seed_155 held out) from 0096 online_4000, with lengths 1-10 weighted 1:1:1:2:2:2:3:3:3:3 so that most episodes hand off within the first few controls and, for the 8-10-control sources, a suffix of 8-10 is the cold reset; evaluated on the three shortest sources and the held-out seed_155, cold reset at every checkpoint. Tests whether the remaining two to four controls to the cold reset close when the stream is devoted to the trajectories that are closest to it.'),
    'cyl_spine172_from96': dict(index=99, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_172', 'seed_116', 'seed_174'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9], suffix_weights=[1, 1, 1, 2, 2, 2, 2, 2, 2], demonstration_suffix_rows=12, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_shared_lambda=50., spine_source='seed_172', start_prefix_set=[0, 1, 2, 3, 4], start_prefix_weights=[2, 2, 1, 1, 1], start_prefix_share=.4,
        actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=4000, checkpoints=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0096_cyl_ms_s1to9_long_from94', 'online_4000'],
        hypothesis='Pin the cold reset to one path. At every prefix state from 1 on, the 0096 online_4000 actor is within 0.02-0.2 of the recorded control of the short sources, but at the cold reset it is 1.4-2.0 from every source first control: the cold reset is the one state all training sources share and their first controls differ, so the actor regresses toward their mean. The 8-control source seed_172 becomes the spine: every one of its rows is a supervised, per-state pinned row (actor_bc_shared_lambda 50; the other sources first-control rows are excluded from BC by the shared-state rule), 40 % of the episodes hand off at its prefixes 0-4 (weights 2:2:1:1:1) and the rest keep the long-weighted suffix stream over all 54 sources; evaluated on the three shortest sources, the held-out trio and the cold reset at every checkpoint. Success criterion unchanged: a frozen cold-reset turn.'),
    'cyl_spine172_longrl_from99': dict(index=100, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_172', 'seed_116', 'seed_174'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9], suffix_weights=[1, 1, 1, 2, 2, 2, 2, 2, 2], demonstration_suffix_rows=12, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_shared_lambda=50., spine_source='seed_172', start_prefix_set=[0, 1, 2, 3, 4], start_prefix_weights=[3, 3, 2, 1, 1], start_prefix_share=.6,
        actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=40000, checkpoints=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0099_cyl_spine172_from96', 'online_4000'],
        hypothesis='Long cold-reset reinforcement learning on the cylinder, the one lever not yet tried there: ten times the interactions of every earlier stage (40,000; the hexagon run bootstrap_0004 stopped at 53,800 with no success) from the 0099 online_4000 actor, which completes the turn from the states after seed_172 recorded controls 2-4 in 6/30, 7/30 and 22/30 trials and reached one changing-policy cold-reset success in 97 episodes. 60 % of the episodes start at the spine prefixes 0-4 with weights 3:3:2:1:1 (the cold reset and the state after one control carry half of them), the rest keep the long-weighted suffix stream over the 51 training sources; same supervision as 0099. cylinder_first_control_0001 section 4 shows that closed-loop strategies tolerant of 0.03-0.10 execution error exist in this task, so the question is whether SAC can find one from a policy that drops at its third control. Evaluated on the three shortest sources, the held-out trio, the shared cells k0-k4 and the cold reset at every 5,000 interactions. Success criterion unchanged: a frozen cold-reset turn.'),
    'cyl_spine172_longrl_alphafreeze_from99': dict(index=101, **REC, prefix_controls=6, prefix_set=[8], prefix_sources=['cylinder_demonstrations_0001', 120], holdout_source_names=['seed_153', 'seed_154', 'seed_155'], eval_source_names=['seed_172', 'seed_116', 'seed_174'],
        suffix_set=[1, 2, 3, 4, 5, 6, 7, 8, 9], suffix_weights=[1, 1, 1, 2, 2, 2, 2, 2, 2], demonstration_suffix_rows=12, actor_bc_rows='per_item', actor_bc_last_rows=8, actor_bc_shared_lambda=50., spine_source='seed_172', start_prefix_set=[0, 1, 2, 3, 4], start_prefix_weights=[3, 3, 2, 1, 1], start_prefix_share=.6,
        actor_bc_lambda=100., actor_bc_std_target=.01, eval_stochastic_trials=3, actor_lr_override=1e-4, online_policy_interactions=40000, checkpoints=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], alpha_freeze=True,
        env_overrides={'object_shape': 'circular'}, eval_recorded_prefixes=False, eval_cold_every_point=True, initial_from=['bootstrap_variant_0099_cyl_spine172_from96', 'online_4000'],
        hypothesis='Replicate of 0100 with the temperature frozen at its inherited value. In 0100 the learned temperature rose from 0.027 to 1.39 over 40,000 interactions because the policy entropy (-82 to -51) never approaches the -16 target under the 0.01 spread anchor; the critic value at the supervised row drifted from -6.6 to -106, the actor BC loss grew 44-fold and the stream reward turned negative after 24,000 interactions, while the cold reset stayed at 0 successes in 2,698 training episodes and 0/5 at every checkpoint. Freezing alpha (template knob alpha_freeze, as in the wave-3b hexagon stages) removes that drift so that the long cold-reset run tests exploration rather than temperature runaway. Everything else as 0100. Success criterion unchanged: a frozen cold-reset turn.'),
}
SAC0004_SEEDS = dict(train_seed=17491, demonstration_seed=17492, prefix_seed=17493, handoff_noise_seed=17494,
                     by_prefix={'6': list(range(17501, 17506)) + list(range(22061, 22066)), '7': list(range(17521, 17526)) + list(range(22071, 22076)), '8': list(range(17531, 17536)) + list(range(22081, 22086)),
                                **{str(p): list(range(17561 + 10 * p, 17566 + 10 * p)) + list(range(22001 + 10 * p, 22006 + 10 * p)) for p in range(1, 6)}},
                     noisy_by_prefix={'7': list(range(17541, 17546)), '8': list(range(17551, 17556)),
                                      **{str(p): list(range(17801 + 10 * p, 17806 + 10 * p)) for p in range(1, 7)}},
                     cold=list(range(17511, 17516)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy(source, local, expected, bindings, D, **extra):
    assert sha(source) == expected, source
    (D / local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, D / local)
    bindings[local] = dict(source=str(source), sha256=expected, **extra)


def fresh_seeds(index):
    base = 19000 + 10 * index
    return dict(train_seed=base, demonstration_seed=base + 1, prefix_seed=base + 2, handoff_noise_seed=base + 3,
                by_prefix={str(p): [20000 + 100 * index + 10 * p + i for i in range(1, 6)] + [24000 + 100 * index + 10 * p + i for i in range(1, 6)] for p in range(1, 9)},
                noisy_by_prefix={str(p): [21000 + 100 * index + 10 * p + i for i in range(1, 6)] for p in range(1, 9)},
                cold=[20000 + 100 * index + 90 + i for i in range(1, 6)])


def collect_seed_values(value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            if 'seed' in k.lower() and isinstance(v, (int, list, dict)):
                collect_seed_values(v, out) if isinstance(v, (list, dict)) else out.add(v)
            elif isinstance(v, (dict, list)):
                collect_seed_values(v, out)
    elif isinstance(value, list):
        for v in value:
            out.add(v) if isinstance(v, int) else collect_seed_values(v, out)


def used_seeds(exclude_dir):
    used = set()
    for proto in A.glob('*/protocol.json'):
        if proto.parent == exclude_dir or proto.parent == P:
            continue
        collect_seed_values(json.loads(proto.read_text()), used)
    return used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--allow-runner-change', action='store_true', help='permit stamping with a runner digest that differs from earlier stamped variants')
    args = parser.parse_args()
    if args.list or not args.variant:
        for name, spec in VARIANTS.items():
            print(f"{spec['index']:02d} {name}: {spec['hypothesis']}")
        return
    assert PINS and all(sha(T / name) == digest for name, digest in PINS.items()), 'template files differ from their pins'
    siblings = [d for d in A.glob('bootstrap_variant_[0-9]*') if (d / 'run.py').is_file()]
    assert args.allow_runner_change or all(sha(d / 'run.py') == PINS['run.py'] for d in siblings), 'an earlier stamped variant carries a different runner; pass --allow-runner-change deliberately'
    spec = VARIANTS[args.variant]
    settings = dict(BASE, **{k: v for k, v in spec.items() if k not in ('index', 'hypothesis')})
    D = A / f"bootstrap_variant_{spec['index']:04d}_{args.variant}"
    if D.exists():
        raise SystemExit(f'{D} already exists; variants are never overwritten')
    s = SAC0004_SEEDS if settings['seed_matched'] else fresh_seeds(spec['index'])
    eval_prefixes = sorted((set(settings['prefix_set']) | {6, 7, 8}) - {0}) if settings['eval_recorded_prefixes'] else []  # the recorded hexagon prefixes cannot be replayed in an overridden environment
    assert settings['env_overrides'] is None or (not settings['eval_recorded_prefixes'] and settings['prefix_sources'] is not None), 'an environment override needs multi-source demonstrations recorded in that environment and no recorded-prefix evaluation'
    noisy_prefixes = sorted(settings['handoff_noise_prefixes']) if settings['handoff_noise_sigma'] > 0 else []
    protocol_seeds = {s['train_seed'], s['demonstration_seed'], s['prefix_seed'], s['handoff_noise_seed'], *s['cold'],
                      *[v for p in eval_prefixes for v in s['by_prefix'][str(p)][:settings['eval_stochastic_trials']]], *[v for p in noisy_prefixes for v in s['noisy_by_prefix'][str(p)]]}
    if not settings['seed_matched']:
        clash = protocol_seeds & used_seeds(D)
        assert not clash, f'fresh seeds collide with earlier protocols: {sorted(clash)}'
    D.mkdir()
    bindings = {}
    prior_execution = json.loads((P / 'execution_manifest.json').read_text())
    prior_manifest = json.loads((P / 'manifest.json').read_text())
    assert all(prior_manifest.get(name, sha(P / name)) == digest for name, digest in prior_execution.items())
    sources = {name: digest for name, digest in prior_execution.items() if name.startswith('source/')}
    assert sources and all(sha(P / name) == digest for name, digest in sources.items())
    shutil.copytree(P / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copied = {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()}
    assert copied == set(sources) and all(sha(D / name) == digest for name, digest in sources.items())
    bindings['source/'] = dict(source=str(P / 'source'), files=len(sources), inherited_execution_manifest_sha256=sha(P / 'execution_manifest.json'))
    learner_files = ('initial.pt', 'initial.pt.metadata.json', 'inherited_state.json', 'prior_replay_lineage_only.npz')
    for name in ('source_config.json', 'initial_grasp.npz', 'prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json',
                 'mpc_correction_pairs.npz', 'mpc_correction_pairs.metadata.json', 'mpc_recovery_transitions.npz'):
        copy(P / name, name, prior_execution[name], bindings, D, inherited_from='bootstrap_suffix_sac_0004')
    if settings['initial_from'] is None:
        for name in learner_files:
            copy(P / name, name, prior_execution[name], bindings, D, inherited_from='bootstrap_suffix_sac_0004')
        inherited_checkpoint = 'bootstrap_suffix_sac_0001/checkpoints/online_2000'
        lineage = dict(rows=2808, open_episode=True)
        inherited_actor_lr = 3e-4
    else:
        src_dir, label = settings['initial_from']
        S = A / src_dir / 'checkpoints' / label
        src_manifest = json.loads((S / 'manifest.json').read_text())
        summary_path = A / src_dir / 'summary.json'
        if summary_path.exists():
            assert json.loads(summary_path.read_text())['status'] == 'complete_sac_stage', 'inherit only from a completed variant stage'
        else:
            # A committed checkpoint of a still-running stage: its directory is complete and hash-listed once renamed.
            assert (S / 'manifest.json').exists() and (S / 'state.json').exists()
        for name, local in (('agent.pt', 'initial.pt'), ('agent.pt.metadata.json', 'initial.pt.metadata.json'),
                            ('state.json', 'inherited_state.json'), ('replay.npz', 'prior_replay_lineage_only.npz')):
            copy(S / name, local, src_manifest[name], bindings, D, inherited_from=f'{src_dir}/checkpoints/{label}')
        if summary_path.exists():
            copy(A / src_dir / 'summary.json', 'inputs/inherited_stage/summary.json', sha(A / src_dir / 'summary.json'), bindings, D)
        else:
            bindings['inputs/inherited_stage/summary.json'] = dict(source=str(summary_path), absent_at_stamp_time=True, inherited_stage_still_running=True)
        copy(A / src_dir / 'protocol.json', 'inputs/inherited_stage/protocol.json', sha(A / src_dir / 'protocol.json'), bindings, D)
        if (A / src_dir / 'findings.md').exists():
            copy(A / src_dir / 'findings.md', 'inputs/inherited_stage/findings.md', sha(A / src_dir / 'findings.md'), bindings, D)
        else:
            bindings['inputs/inherited_stage/findings.md'] = dict(source=str(A / src_dir / 'findings.md'), absent_at_stamp_time=True)
        inherited_checkpoint = f'{src_dir}/checkpoints/{label}'
        st = json.loads((D / 'inherited_state.json').read_text())
        src_protocol = json.loads((A / src_dir / 'protocol.json').read_text())
        inherited_actor_lr = float(src_protocol.get('actor_lr_override') or src_protocol.get('inherited_actor_lr') or 3e-4)
        lineage = dict(rows=st['replay_rows'], open_episode=bool(st['unfinished_episode']))
    for name in ('findings.md', 'summary.json', 'protocol.json'):
        copy(P / name, 'inputs/sac0004/' + name, prior_manifest[name], bindings, D)
    for name in ('run.py', 'test_protocol.py'):
        copy(T / name, name, PINS[name], bindings, D, template=str(T), pinned=True)
    copy(T / 'prepare.py', 'inputs/template/prepare.py', sha(T / 'prepare.py'), bindings, D, template=str(T), pinned=False)
    copy(T / 'README.md', 'inputs/template/README.md', sha(T / 'README.md'), bindings, D, template=str(T), pinned=False)
    assert settings['initial_from'] is not None or sha(D / 'initial.pt') == 'e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892'
    assert sha(D / 'prefix_transitions.npz') == '7b75bef17b559652bc7733b081dc03e4edaf012f46bfd0d11a349b5d4d816d8f'
    inherited = json.loads((D / 'inherited_state.json').read_text())
    counters = inherited['learner_counters']
    if settings['initial_from'] is None:
        assert counters == dict(bc_updates=12000, pretrain_updates=500, online_updates=2000)
    steps = inherited.get('optimizer_steps', dict(critic=2500, actor=2000, alpha=2000))  # SAC0001's state.json predates this field
    inherited_steps = {k: (v[0] if isinstance(v, list) else v) for k, v in steps.items()}
    n_online = settings['online_policy_interactions']
    assert settings['checkpoints'][-1] == n_online and all(0 < c <= n_online for c in settings['checkpoints'])
    multisource = None
    assert settings['actor_bc_rows'] != 'per_item' or settings['prefix_sources'] is not None, "actor_bc_rows='per_item' is defined for multi-source stages only"
    assert settings['actor_bc_shared_lambda'] is None or settings['actor_bc_rows'] == 'per_item', 'actor_bc_shared_lambda needs per_item rows'
    assert settings['spine_source'] is None or settings['start_prefix_set'], 'a spine needs start_prefix_set'
    assert (settings['actor_bc_shared_lambda'] is None and settings['spine_source'] is None) or settings['actor_bc_shared_prefix_rows'], 'shared-row supervision is switched off by actor_bc_shared_prefix_rows=False'
    assert settings['prefix_sources'] is not None or (settings['start_prefix_set'] is None and settings['start_prefix_share'] == 0), 'shared cold-start prefixes need prefix_sources'
    if settings['prefix_sources'] is not None:
        collections = [tuple(settings['prefix_sources'])] + [tuple(x) for x in settings['extra_sources']]
        names, entries, skipped_labelless = [], [], []
        for coll_dir, max_sources in collections:
            C = A / coll_dir
            coll_names = sorted(q.name for q in (C / 'episodes').iterdir() if (q / 'report.json').exists())[:max_sources]
            for name in coll_names:
                rep_ = json.loads((C / 'episodes' / name / 'report.json').read_text())
                assert rep_['strict_force_success'] and rep_['stored'] and rep_['independent_audit']['bitexact_replay']
                if rep_.get('policy_controls', 0) and rep_.get('mpc_controls', 0) == 0:
                    skipped_labelless.append(name); continue  # a labelled branch whose perturbed/policy control ended the turn carries no correction rows
                if settings['max_source_controls'] is not None and rep_['controls'] > settings['max_source_controls']:
                    continue  # source-length filter (e.g. a short-trajectory stage whose long suffixes reach the cold reset)
                (D / 'sources' / name).mkdir(parents=True)
                for f in ('transitions.npz', 'physics.npz', 'report.json'):
                    copy(C / 'episodes' / name / f, f'sources/{name}/{f}', sha(C / 'episodes' / name / f), bindings, D, demonstration_source=str(C))
                entries.append(dict(name=name, transitions=f'sources/{name}/transitions.npz', physics=f'sources/{name}/physics.npz', sha256=sha(D / f'sources/{name}/transitions.npz'), controls=rep_['controls'], collection=coll_dir, policy_rows=int(rep_.get('policy_controls', 0)),
                                    demonstration_rows_available=int(rep_['mpc_controls'] if rep_.get('policy_controls', 0) else rep_['controls'])))  # DAgger: only the MPC segment after the last policy control
                names.append(name)
        assert len(names) >= 2, 'multi-source stages need at least two verified demonstrations'
        m = int(settings['demonstration_suffix_rows'] or max(settings['suffix_set']))
        assert m >= max(settings['suffix_set'])
        holdouts = list(settings['holdout_source_names']) if settings['holdout_source_names'] else (names[len(names) - settings['holdout_source_count']:] if settings['holdout_source_count'] else [])
        assert set(holdouts) <= set(names) and set(settings['eval_source_names'] or []) <= set(names) - set(holdouts)
        training = [n_ for n_ in names if n_ not in holdouts]
        assert len(training) >= 2
        if settings['suffix_weights'] is not None:
            assert len(settings['suffix_weights']) == len(settings['suffix_set'])
        shared_groups, shared_depth = {}, {}
        if settings['start_prefix_set']:
            # Shared cold-start prefixes: for each k the largest set of training sources whose first k controls are
            # bit-identical (k = 0: every training source, the cold reset itself). Verified on the copied files.
            assert 0 < settings['start_prefix_share'] <= 1 and (settings['start_prefix_weights'] is None or len(settings['start_prefix_weights']) == len(settings['start_prefix_set']))
            loaded = {}
            for name in training:
                with np.load(D / f'sources/{name}/transitions.npz', allow_pickle=False) as z:
                    loaded[name] = (z['actions'].copy(), z['obs'].copy())
            spine = settings['spine_source']
            assert spine is None or spine in training, 'the spine must be a training source'
            # Labelled sources (DAgger / tube branches) copy a demonstration's prefix bit-for-bit; they are never members of a
            # shared group (their supervised rows are the correction segment only) and never representatives.
            policy_rows_of = {e['name']: e.get('policy_rows', 0) for e in entries}
            unlabelled = [n_ for n_ in training if not policy_rows_of.get(n_, 0)]
            assert spine is None or not policy_rows_of.get(spine, 0)
            for k in settings['start_prefix_set']:
                if k == 0:
                    group = list(unlabelled)
                else:
                    buckets = {}
                    for name in unlabelled:
                        if len(loaded[name][0]) > k:
                            buckets.setdefault(loaded[name][0][:k].tobytes(), []).append(name)
                    if spine is None:
                        group = max(buckets.values(), key=len)
                        assert len(group) >= 2, f'no training sources share their first {k} controls'
                    else:
                        # Spine curriculum: the shared path is the spine's own path; the group is every source whose
                        # first k controls coincide with it (the spine alone once the others have diverged).
                        assert len(loaded[spine][0]) > k, f'spine shorter than start prefix {k}'
                        group = buckets[loaded[spine][0][:k].tobytes()]
                    ref_a, ref_o = loaded[group[0]]
                    for name in group:
                        assert np.array_equal(loaded[name][0][:k], ref_a[:k]) and np.array_equal(loaded[name][1][:k + 1], ref_o[:k + 1])
                group = sorted(group)
                if spine is not None:
                    assert spine in group; group.remove(spine); group.insert(0, spine)  # the spine is the evaluation representative
                shared_groups[str(k)] = group
                for name in group:
                    shared_depth[name] = max(shared_depth.get(name, 0), int(k))
            if spine is not None:
                shared_depth[spine] = int(len(loaded[spine][0]))  # every spine row is a supervised row (no other source labels its states)
            ks = sorted(int(k) for k in settings['start_prefix_set'])
            for k in ks:
                if k >= 1:
                    assert set(shared_groups[str(k)]) <= set(shared_groups[str(k - 1)] if str(k - 1) in shared_groups else training), 'shared groups must nest'
            for name, d in shared_depth.items():
                assert d == 0 or loaded[name][0].shape[0] <= m, 'a source with shared prefix rows must be buffered whole'
            # Longest prefix each source shares with any other training source: rows up to that index lie at states another
            # source also visits (index 0, the cold reset, for every source); the runner never uses such rows as BC targets
            # unless they are shared-group rows (index < shared_prefix_depth), whose labels agree across the group.
            def lcp(a, b):
                j = 0
                while j < min(len(a), len(b)) and np.array_equal(a[j], b[j]):
                    j += 1
                return j
            shared_state_depth = {name: max(lcp(loaded[name][0], loaded[other][0]) for other in unlabelled if other != name) for name in unlabelled}
            shared_state_depth.update({name: -1 for name in training if name not in unlabelled})  # correction rows lie at states no other source visits
            if not settings['actor_bc_shared_prefix_rows']:
                shared_depth = {}
        else:
            shared_state_depth = {}
            assert settings['start_prefix_share'] == 0 and settings['actor_bc_shared_prefix_rows'], 'shared-prefix settings need start_prefix_set'
        multisource = dict(prefix_sources=entries, prefix_source_names=training, holdout_source_names=holdouts, skipped_labelless_sources=skipped_labelless, suffix_set=list(settings['suffix_set']), suffix_weights=settings['suffix_weights'],
                           demonstration_suffix_rows=m, eval_source_names=(list(settings['eval_source_names']) if settings['eval_source_names'] else training[:settings['eval_source_count']]) + holdouts,
                           eval_stochastic_seeds_by_suffix={str(s_): [26000 + 100 * spec['index'] + 10 * s_ + i for i in range(1, settings['eval_stochastic_trials'] + 1)] for s_ in settings['suffix_set']},
                           eval_noisy_seeds_by_suffix={str(s_): [27000 + 100 * spec['index'] + 10 * s_ + i for i in range(1, 6)] for s_ in settings['suffix_set']},
                           handoff_noise_suffixes=list(settings['handoff_noise_suffixes']),
                           start_prefix_set=settings['start_prefix_set'], start_prefix_weights=settings['start_prefix_weights'], start_prefix_share=settings['start_prefix_share'],
                           shared_prefix_groups=shared_groups, shared_prefix_group_sizes={k: len(v) for k, v in shared_groups.items()}, shared_prefix_depth=shared_depth, shared_state_depth=shared_state_depth,
                           eval_shared_seeds_by_k={str(k_): [28000 + 100 * spec['index'] + 10 * k_ + i for i in range(1, settings['eval_stochastic_trials'] + 1)] for k_ in (settings['start_prefix_set'] or []) if k_ >= 1},
                           actor_bc_last_rows=settings['actor_bc_last_rows'], actor_bc_shared_prefix_rows=settings['actor_bc_shared_prefix_rows'],
                           spine_source=settings['spine_source'], actor_bc_shared_lambda=settings['actor_bc_shared_lambda'], online_demo_sarsa=settings['online_demo_sarsa'], labelled_handoff_share=settings['labelled_handoff_share'],
                           multisource_note='per episode one verified cold-reset MPC demonstration is drawn uniformly and its first (length - suffix) controls are physically replayed and verified bit-for-bit; the demonstration buffer holds the last demonstration_suffix_rows rows of every source; evaluations run per eval source and suffix length plus the recorded prefixes in eval_prefixes (none for an environment-override stage) and cold reset')
    write(D / 'input_bindings.json', bindings)
    config = json.loads((D / 'source_config.json').read_text())
    config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
    if settings['env_overrides']:
        config['environment'].update(settings['env_overrides'])  # route (b): a disclosed deviation of the archived environment (e.g. object_shape)
        try:
            for e in entries:  # every demonstration must have been recorded in exactly the stage environment (grasp file and render size aside)
                rec_env = json.loads((D / f"sources/{e['name']}/report.json").read_text()).get('environment_config') or {}
                for k, v in config['environment'].items():
                    if k in ('initial_grasp', 'render_width', 'render_height'): continue
                    assert k in rec_env and rec_env[k] == v, f"{e['name']}: demonstration recorded with {k}={rec_env.get(k)} but the stage environment has {k}={v}"
        except AssertionError:
            shutil.rmtree(D); raise
    write(D / 'config.json', config)
    start, end = settings['demonstration_recorded_rows']
    rows_recorded = end - start
    rows_mpc = 5 if settings['demonstration_include_mpc'] else 0
    if multisource is not None:
        rows_recorded, rows_mpc = 0, 0
    fr_pre, fr_on = settings['pretrain_offline_fraction'], settings['online_offline_fraction']
    n_pre, n_on = int(round(256 * fr_pre)), int(round(256 * fr_on))
    overlap = sorted(set(range(start, end)) & set(range(0, max(settings['prefix_set']))))
    if settings['actor_bc_lambda'] <= 0:
        bc_term_text = 'none'
    else:
        rows_text = {'first': 'the first row of each demonstration episode', 'first_k': f"the first {settings['actor_bc_first_rows']} rows of each demonstration episode", 'per_item': f"the last {settings['actor_bc_last_rows']} rows of each demonstration episode plus its shared cold-start prefix rows" + (f"; additionally, per distinct shared state, lambda_shared {settings['actor_bc_shared_lambda']} * (MSE of the actor mean to the label + MSE of the log-std to log(std target)) summed over the distinct shared states (one term per state, however many sources sit on it)" if settings['actor_bc_shared_lambda'] else '')}.get(settings['actor_bc_rows'], 'all demonstration rows')
        bc_term_text = f"lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over {rows_text}, added to the single SAC actor step"
    write(D / 'protocol.json', dict(
        schema_version=2, kind='diagnostic-prefix-assisted-suffix-sac-variant',
        variant=args.variant, variant_index=spec['index'], hypothesis=spec['hypothesis'],
        status='frozen-ready-for-separate-explicit-execution',
        settings_differing_from_sac0004={k: v for k, v in settings.items() if BASE.get(k) != v},
        prefix_controls=settings['prefix_controls'], prefix_set=settings['prefix_set'], prefix_weights=settings['prefix_weights'], prefix_seed=s['prefix_seed'],
        prefix_sampling=('uniform' if settings['prefix_weights'] is None else f"weighted {settings['prefix_weights']}") + ' over prefix_set per episode with a dedicated numpy Generator (row-level mix is episode-length weighted); every exact prefix control is physically replayed and verified bit-for-bit against the recorded source',
        handoff_noise_sigma=settings['handoff_noise_sigma'], handoff_noise_prefixes=noisy_prefixes, handoff_noise_seed=s['handoff_noise_seed'],
        handoff_noise='when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise"' if noisy_prefixes else 'off',
        fixed_prefix_stage=len(settings['prefix_set']) == 1 and not noisy_prefixes, automatic_stage_advancement=False,
        motivation='SAC0003/SAC0004 (fixed prefix 6, 8 demonstration rows mixed 30/70) produced no frozen success and lost prefix-7 retention by online 500 even with correct demonstration values. Read-only probes (review workflow, 2026-09-25) found the MPC continuation unfinishable by this actor, a narrow success basin around recorded control 7, and critic-led forgetting without prefix-7 rows. This family varies named factors from the same SAC0001 checkpoint.',
        seed_matched_to_sac0004=settings['seed_matched'],
        train_seed=s['train_seed'], demonstration_seed=s['demonstration_seed'], physical_reset_seed=42,
        warmup_complete_episodes=256,
        learning_gate='Verified off-policy demonstration rows in the fixed demonstration buffer; warm-up policy successes are recorded but not required.',
        demonstration_include_mpc=settings['demonstration_include_mpc'], demonstration_recorded_rows=[start, end],
        demonstration_rows=(sum(min(multisource['demonstration_suffix_rows'], e['demonstration_rows_available']) for e in multisource['prefix_sources'] if e['name'] in multisource['prefix_source_names'])) if multisource else rows_mpc + rows_recorded, demonstration_rows_mpc=rows_mpc, demonstration_rows_recorded=rows_recorded,
        demonstration_episodes=len(multisource['prefix_source_names']) if multisource else int(settings['demonstration_include_mpc']) + 1,
        **(multisource or {}),
        demonstration_sources=dict(mpc='mpc_correction_pairs.npz rows 0-4: actual corrected native MPC controls 7-11 of the recovery episode, terminal success' if settings['demonstration_include_mpc'] else 'excluded',
                                   recorded=f'prefix_transitions.npz rows {start}-{end - 1}: recorded BC-origin source controls {start + 1}-{end}, terminal success'),
        demonstration_and_prefix_overlap=(f'demonstration rows {overlap} are the same source transitions that prefix replay re-executes for prefixes above {overlap[0]}; they are present once in the demonstration buffer and never appended to the online replay' if overlap else 'none: no demonstration row is a prefix-replayed transition of any prefix in prefix_set'),
        collected_prefix_rows_policy='physically replayed and perturbed prefix controls of collected episodes are never appended to the online replay and never count as policy interactions',
        pretrain_offline_fraction=fr_pre, online_offline_fraction=fr_on,
        mixing=f'runner mixed_batch_with_origin reproducing mpc_dex.replay.sample_mixed draw order with a per-row origin mask: pretraining {n_pre} demonstration + {256 - n_pre} online rows per batch; online {n_on} + {256 - n_on}',
        actor_bc_lambda=settings['actor_bc_lambda'], actor_bc_rows=settings['actor_bc_rows'], actor_bc_first_rows=settings['actor_bc_first_rows'],
        actor_lr_override=settings['actor_lr_override'],
        actor_bc_std_target=settings['actor_bc_std_target'], actor_bc_std_lambda=settings['actor_bc_std_lambda'] if settings['actor_bc_std_lambda'] is not None else settings['actor_bc_lambda'],
        actor_bc_std_term=(f"lambda_std * MSE(actor log-std at the supervised rows, log({settings['actor_bc_std_target']})) added to the same actor step" if settings['actor_bc_std_target'] is not None else 'none'),
        actor_bc_term=bc_term_text,
        actor_lr=f'inherited {inherited_actor_lr}' if settings['actor_lr_override'] is None else f"overridden to {settings['actor_lr_override']} for this stage (inherited {inherited_actor_lr}; ramp fraction unchanged)",
        actor_on_demonstration_states=settings['actor_on_demonstration_states'],
        actor_loss_rows='all batch rows' if settings['actor_on_demonstration_states'] else 'on-policy rows only (SAC actor and temperature losses; logged entropy uses the same rows); critic step unchanged',
        pretrain='entropy_masked_pretrain_step on mixed batches: on-policy rows unchanged SoftSARSA with pretrain alpha .0001; demonstration rows entropy-free; sampled raw rewards / 200',
        online=('runner_update replicating SACAgent.update call-for-call (bit-identical when actor_bc_lambda=0 and actor_on_demonstration_states=true) with the protocol additions above; sampled raw rewards / 200; one update per actual policy step; cumulative LR ramp'+ (' ; with online_demo_sarsa the demonstration rows of every online batch bootstrap along their recorded next action without the entropy term (the pretraining target kept online) while on-policy rows keep the standard backup' if settings['online_demo_sarsa'] else '')),
        critic_pretrain_updates=500, online_policy_interactions=n_online, checkpoints=settings['checkpoints'],
        inherited_checkpoint=inherited_checkpoint, prior_replay_lineage=lineage, inherited_actor_lr=inherited_actor_lr, env_overrides=settings['env_overrides'],
        env_override_note=(None if not settings['env_overrides'] else 'the environment differs from the archived task in the listed fields (route b); the inherited checkpoint was trained on the archived task; every demonstration source was recorded and audited in the overridden environment; the recorded-prefix evaluations are disabled'),
        inherited_counters=counters,
        expected_final_counters=dict(bc_updates=counters['bc_updates'], pretrain_updates=counters['pretrain_updates'] + 500, online_updates=counters['online_updates'] + n_online),
        inherited_optimizer_steps=inherited_steps,
        expected_final_optimizer_steps=dict(critic=inherited_steps['critic'] + 500 + n_online, actor=inherited_steps['actor'] + n_online, alpha=inherited_steps['alpha'] + (0 if settings['alpha_freeze'] else n_online)),
        alpha_freeze=settings['alpha_freeze'], alpha_override=settings['alpha_override'],
        temperature=('frozen at ' + ('the override value' if settings['alpha_override'] is not None else 'the inherited value') + ' for this stage; alpha loss logged only, alpha optimizer untouched') if settings['alpha_freeze'] else ('learned, starting from override ' + str(settings['alpha_override']) if settings['alpha_override'] is not None else 'learned from the inherited value'),
        batch_size=256, utd=1, replay_capacity=100000,
        raw_reward_divisor=200., inherited_learned_alpha=inherited['normalized_alpha'],
        pretrain_alpha=.0001, actor_warmup_updates=5000,
        eval_deterministic_seed=42, eval_prefixes=eval_prefixes,
        eval_stochastic_seeds_by_prefix={str(p): s['by_prefix'][str(p)][:settings['eval_stochastic_trials']] for p in eval_prefixes},
        eval_noisy_prefixes=noisy_prefixes, eval_noisy_seeds_by_prefix={str(p): s['noisy_by_prefix'][str(p)] for p in noisy_prefixes},
        eval_final_cold_stochastic_seeds=s['cold'], eval_cold_stochastic_every_point=0 in settings['prefix_set'] or 0 in (settings['start_prefix_set'] or []) or settings['eval_cold_every_point'],
        evaluation_points=['pretraining_baseline'] + [f'online_{c}' for c in settings['checkpoints']],
        evaluation='each point: for every prefix in eval_prefixes one deterministic and five stochastic trials; for every noisy prefix five stochastic trials whose final prefix control is perturbed with a Generator seeded by the trial seed; cold deterministic; five cold stochastic trials at the final point (every point when 0 is in prefix_set or in start_prefix_set, or when eval_cold_every_point is set); for every shared cold-start prefix length k >= 1 in start_prefix_set one deterministic and eval_stochastic_trials stochastic trials from the state after the k shared controls (replayed from the group representative); evaluations restore training RNG and never enter replay',
        physical_horizon_including_prefix=100, policy_interface={'kind': 'absolute_joint_target'},
        learner_import=f'Full exact network, target, optimizer, counter, learned alpha and learned std import from {inherited_checkpoint}; no resets',
        replay='Fresh empty online replay for actual new-stage policy suffix transitions; separate immutable demonstration buffer; SAC0001 replay copied as lineage only',
        prior_open_replay_semantics='Preserve source open row unchanged; never import, finish, discard in source, or connect it to a new reset',
        sampling=f"Singleton batch (1,60) CUDA float32 actor.sample; direct sampled log probability; global seed {s['train_seed']}",
        demonstration_calibration='Critic values on the demonstration rows versus entropy-free discounted raw returns / 200, deterministic actor-mean distance to each demonstration episode\'s first action (flagged when that row is the prefix-6 handoff), actor log density of demonstration actions, at baseline, after pretraining and at each online checkpoint',
        normalization_caveat='Reward/entropy units remain those of SAC0001; inherited learned alpha is not reset. Warm-up calibration pools rows of all collected prefixes; only the held-out prefix-6 set is comparable across variants.',
        complete_episode_definition='Environment terminated or horizon truncated; any truncated return tail remains unknown',
        warmup_alpha_for_calibration=inherited['normalized_alpha'],
        pretrain_calibration_alpha_caveat='Report inherited learned online alpha separately from unchanged .0001 SoftSARSA objective.',
        no_trajectory_export=True, eligible_for_primitive=False, main_mpc_dataset_eligible=False,
        new_bc_updates=0, head_refinement=False, extra_epochs=False,
        post_success_continuation=False, resume_implementation=False,
        checkpoint_semantics='New curriculum stage with fresh reset/replay/RNG; full learner continuation, not exact environment resume',
        future_checkpoint_use='A later trained 60D checkpoint requires a new frozen cold-policy collection; assisted and demonstration-mixed episodes are never primitive evidence.',
        template_run_sha256=PINS['run.py'], source_bindings=bindings))
    (D / 'README.md').write_text(f"""# Variant {spec['index']:02d} `{args.variant}`; prepared, not executed

{spec['hypothesis']}

Settings that differ from the SAC0004 baseline: `{json.dumps({k: v for k, v in settings.items() if BASE.get(k) != v})}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds {'match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide)' if settings['seed_matched'] else 'are fresh'}. Template: `{T.name}`.

```bash
.venv/bin/python -m pytest -q {D.relative_to(A.parent)}/test_protocol.py
MUJOCO_GL=egl .venv/bin/python {D.relative_to(A.parent)}/run.py --execute > {D.relative_to(A.parent)}/run.log 2>&1
```
""")
    write(D / 'execution_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.name not in EXCLUDED})
    print(json.dumps(dict(variant=args.variant, directory=str(D), prepared=True, frozen=True, executed=False,
                          execution_manifest_sha256=sha(D / 'execution_manifest.json'))))


if __name__ == '__main__':
    main()
