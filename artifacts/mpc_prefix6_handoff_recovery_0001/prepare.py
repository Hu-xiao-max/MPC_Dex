"""Prepare and freeze the prefix-6 handoff MPC recovery stage; never execute physics."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
STAGE = D.parent / 'bootstrap_suffix_sac_0002'
PRIOR = D.parent / 'mpc_common_noise_recovery_0001'
CLOCK = D.parent / 'mpc_clock_actor_recovery_0001'
EXCLUDED_FROM_FREEZE = ('execution_manifest.json', 'run.log')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy(source, local, expected, provenance):
    assert sha(source) == expected, source
    Path(D / local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, D / local)
    provenance[local] = dict(source=str(source), sha256=expected)


def main():
    if (D / 'execution_manifest.json').exists() or (D / 'runtime.json').exists():
        raise SystemExit('This stage is already frozen or started')
    provenance = {}
    # Frozen numerical source: the byte-identical snapshot of the stage that recorded the prefix.
    stage_manifest = json.loads((STAGE / 'manifest.json').read_text())
    stage_execution = json.loads((STAGE / 'execution_manifest.json').read_text())
    sources = {name: digest for name, digest in stage_execution.items() if name.startswith('source/')}
    assert sources and all(stage_manifest[name] == digest for name, digest in sources.items())
    assert all(sha(STAGE / name) == digest for name, digest in sources.items())
    shutil.copytree(STAGE / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copied = {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()}
    assert copied == set(sources) and all(sha(D / name) == digest for name, digest in sources.items())
    provenance['source/'] = dict(source=str(STAGE / 'source'), files=len(sources),
                                 inherited_execution_manifest_sha256=sha(STAGE / 'execution_manifest.json'))
    # Prefix evidence and the gate failure that motivates this stage.
    for name in ('prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json', 'initial_grasp.npz',
                 'stage_binding.json', 'warmup_gate.json', 'summary.json', 'protocol.json', 'findings.md',
                 'execution_manifest.json', 'config.json'):
        # findings.md postdates the stage's runner-written manifest; bind it by its current hash.
        copy(STAGE / name, f'inputs/prefix_stage/{name}', stage_manifest.get(name, sha(STAGE / name)), provenance)
    copy(STAGE / 'manifest.json', 'inputs/prefix_stage/manifest.json', sha(STAGE / 'manifest.json'), provenance)
    assert stage_manifest['prefix_transitions.npz'] == '7b75bef17b559652bc7733b081dc03e4edaf012f46bfd0d11a349b5d4d816d8f'
    assert sha(D / 'inputs/prefix_stage/initial_grasp.npz') == sha(D / 'source/assets/initial_grasp_40mm.npz')
    gate = json.loads((D / 'inputs/prefix_stage/warmup_gate.json').read_text())
    assert gate == dict(passed=False, minimum_force_successes=1, observed_force_successes=0,
                        complete_warmup_episodes=256, screen_success_count_not_used_as_gate=True)
    binding = json.loads((D / 'inputs/prefix_stage/stage_binding.json').read_text())
    assert binding['selected_prefix_controls'] == 6
    # Same measured 40 mm diagnostic PCA prior, planner recipe and audit helper as the earlier recoveries.
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    for name in ('inputs/previous_recovery/prior_transitions.npz', 'inputs/previous_recovery/reference_planner_report.json',
                 'inputs/previous_recovery/diagnostic_prior.npz', 'protocol.json', 'final_report.json', 'findings.md'):
        copy(PRIOR / name, 'inputs/previous_recovery/' + Path(name).name, prior_manifest[name], provenance)
    copy(PRIOR / 'manifest.json', 'inputs/previous_recovery/manifest.json', sha(PRIOR / 'manifest.json'), provenance)
    copy(PRIOR / 'recovery_recipe.py', 'recovery_recipe.py', '59b99e050f8b9e15482eaebeb02d535117c2d064ac4afc984546e53be30691a2', provenance)
    copy(PRIOR / 'audit_helpers.py', 'audit_helpers.py', prior_manifest['audit_helpers.py'], provenance)
    assert provenance['inputs/previous_recovery/prior_transitions.npz']['sha256'] == '1d8229b86ae66ab986db226199626b3012b12200bbd89e929a200efdc2a1a526'
    clock_manifest = json.loads((CLOCK / 'manifest.json').read_text())
    for name in ('findings.md', 'final_report.json', 'protocol.json'):
        copy(CLOCK / name, 'inputs/clock_recovery_reference/' + name, clock_manifest[name], provenance)
    # Native build cache is copied only to avoid a recompile; the loader rewrites its provenance file,
    # so it is excluded from the frozen manifest and its build/binary hashes are recorded at runtime.
    shutil.copytree(CLOCK / 'native_cache', D / 'native_cache')
    write(D / 'input_provenance.json', provenance)
    config = json.loads((D / 'inputs/prefix_stage/config.json').read_text())
    config['environment']['initial_grasp'] = str(D / 'inputs/prefix_stage/initial_grasp.npz')
    write(D / 'config.json', config)
    write(D / 'protocol.json', dict(
        schema_version=1, kind='diagnostic-prefix6-handoff-mpc-recovery',
        status='frozen-ready-for-separate-explicit-execution',
        purpose='One bounded corrected-native-MPC recovery from the exact prefix-6 handoff state that the trained SAC0001 checkpoint failed in 262 physical attempts; diagnostic non-policy correction labels only',
        motivation='bootstrap_suffix_sac_0002 warm-up gate failed with 0/256 force-qualified successes; the frozen policy cannot supply positive prefix-6 examples',
        prefix_controls=6, prefix_source='inputs/prefix_stage/prefix_transitions.npz',
        prefix_source_sha256=stage_manifest['prefix_transitions.npz'],
        prefix_physics_sha256=stage_manifest['prefix_physics.npz'],
        prefix_verification='every recorded field including full integration state and all 300 2 ms physics samples must match bit-for-bit before handoff',
        reset_seed=42, planner_seed=42, max_mpc_controls=20, action_bound=.98, workers=8,
        cem=dict(samples=350, elites=10, iterations=3, knots=6, horizon_seconds=.5, timestep_seconds=.1, control_seconds=.1),
        cost=dict(position_weight=1000., principal_weight=1., perpendicular_weight=100., workspace_radius=.18,
                  rotation_progress='retained', workers=8),
        reference_candidates='archived ProjectedReferenceCEM: warm nominal plus explicitly projected bounded target reference when current actual target lies out of search; live state never reset/clipped',
        stop=['strict environment and independent force-supported full turn', 'environment termination/failure',
              'no feasible plan', '20 added MPC controls'],
        known_nominal_continuation='The recorded controls 7-9 of the same source trajectory already complete the turn from this handoff (source prefix_report success at 9 controls). MPC output is an independent feedback-derived label set, not proof that a positive action exists; the runner reports the distance between its first correction and recorded control 7.',
        post_success_hold=False, alternate_prefix_or_parameter_sweep=False, policy_inference=False,
        training_updates=0, main_demonstrations=0, production_eligible=False, eligible_for_primitive=False,
        correction_pairs='rows after the six prefix controls only; prefix rows are never expert labels; no successor action is fabricated',
        intended_use='Candidate non-policy seed data for a separately reviewed prefix-6 curriculum stage; use requires its own protocol and root review',
        source_recipe_sha256=provenance['recovery_recipe.py']['sha256'],
        source_prior_sha256=provenance['inputs/previous_recovery/prior_transitions.npz']['sha256'],
        stage_gate_sha256=provenance['inputs/prefix_stage/warmup_gate.json']['sha256'],
        input_provenance_sha256=sha(D / 'input_provenance.json')))
    write(D / 'execution_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and 'native_cache' not in p.parts
          and p.name not in EXCLUDED_FROM_FREEZE})
    print(json.dumps(dict(prepared=True, frozen=True, executed=False,
                          prefix_sha256=stage_manifest['prefix_transitions.npz'],
                          execution_manifest_sha256=sha(D / 'execution_manifest.json'))))


if __name__ == '__main__':
    main()
