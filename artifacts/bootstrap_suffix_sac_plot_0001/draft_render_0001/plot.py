"""Plot completed saved results only; no policy inference, training or physics."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

D = Path(__file__).resolve().parent
SOURCE = D.parent / 'bootstrap_suffix_sac_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def qualified(record):
    """Check consistency of saved summary fields, not a new physics audit."""
    conjunction = bool(
        record['environment_success'] and record['terminated'] and not record['truncated']
        and not record['environment_drop'] and not record['independent_drop']
        and not record['independent_palm_support_violation'] and record['no_external_support']
        and record['model_equality_constraints'] == 0
        and record['final_force_bearing_fingers'] >= 2
        and min(record['final_raw_angle'], record['final_qualified_angle'], record['final_force_qualified_angle']) >= 2 * np.pi)
    if conjunction != record['finger_supported_rotation_success']:
        raise ValueError('Contradictory saved force-qualified success fields')
    return conjunction


def main():
    if (D / 'manifest.json').exists():
        raise SystemExit('Final plot generation exists; preserve it')
    if not (SOURCE / 'summary.json').exists() or not (SOURCE / 'manifest.json').exists():
        raise SystemExit('Completed run summary and final source manifest are required; no live plotting')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    snapshots = D / 'inputs'
    snapshots.mkdir(exist_ok=False)
    read_set = {}

    def read(relative):
        path = SOURCE / relative
        digest = sha(path)
        if source_manifest.get(relative) != digest:
            raise ValueError('Source report/hash mismatch: ' + relative)
        dest = snapshots / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        read_set[relative] = digest
        return json.loads(dest.read_text())

    summary = read('summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Run did not complete')
    protocol = read('protocol.json')
    runtime = read('runtime.json')
    read('resolved_learner.json')
    assert protocol['prefix_controls'] == 7 and summary['counts']['online_policy_interactions'] == 2000
    shutil.copyfile(SOURCE / 'manifest.json', snapshots / 'source_manifest.json')

    episodes = []
    for path in sorted((SOURCE / 'episodes' / 'online').glob('*/report.json')):
        report = read(str(path.relative_to(SOURCE)))
        assert report['phase'] == 'online' and report['prefix_controls'] == 7
        assert report['environment_episode_complete'] and report['episode_policy_may_change_between_decisions']
        assert not report['frozen_policy_success_claim']
        episodes.append(report)
    assert [r['episode_number'] for r in episodes] == list(range(len(episodes)))
    assert len(episodes) == summary['counts']['online_assisted_completed_episodes']
    assert sum(qualified(r) for r in episodes) == summary['counts']['online_assisted_force_successes']
    bins = []
    for begin in range(0, len(episodes), 100):
        group = episodes[begin:begin + 100]
        bins.append(dict(first_episode=begin + 1, last_episode=begin + len(group),
                         n=len(group), successes=sum(qualified(r) for r in group),
                         partial_bin=len(group) < 100,
                         first_completed_online_interaction=group[0]['learner_online_updates'],
                         last_completed_online_interaction=group[-1]['learner_online_updates']))

    evaluations = []
    labels = ['pretraining_baseline', 'online_500', 'online_1000', 'online_2000']
    for label in labels:
        saved = read('evaluations/' + label + '/summary.json')
        assert saved['training_rng_restored'] and saved['learner_tensors_and_counters_unchanged']
        reports = saved['reports']

        def select(prefix):
            return [r for r in reports if r['phase'].startswith(prefix)]

        phase = select('phase_stochastic_')
        cold_det = select('cold_deterministic')
        cold_stoch = select('cold_stochastic_')
        retention = select('retention8_stochastic_')
        retention_det = select('retention8_deterministic')
        phase_det = select('phase_deterministic')
        assert len(phase) == 5 and {r['policy_sampling_seed'] for r in phase} == set(range(17101, 17106))
        assert len(phase_det) == len(cold_det) == 1
        assert len(cold_stoch) == (5 if label == 'online_2000' else 0)
        assert len(retention) == (5 if label in ('pretraining_baseline', 'online_2000') else 0)
        assert len(retention_det) == (1 if retention else 0)
        if retention:
            assert {r['policy_sampling_seed'] for r in retention} == set(range(17121, 17126))
        if cold_stoch:
            assert {r['policy_sampling_seed'] for r in cold_stoch} == set(range(17111, 17116))
        for r in phase + phase_det:
            assert r['prefix_controls'] == 7
        for r in retention + retention_det:
            assert r['prefix_controls'] == 8
        for r in cold_det + cold_stoch:
            assert r['prefix_controls'] == 0
        evaluations.append(dict(label=label, online_updates=0 if label == 'pretraining_baseline' else int(label.split('_')[1]),
            phase7_stochastic_successes=sum(qualified(r) for r in phase), phase7_stochastic_n=5,
            phase7_deterministic_successes=sum(qualified(r) for r in phase_det), phase7_deterministic_n=1,
            retention8_stochastic_successes=sum(qualified(r) for r in retention), retention8_stochastic_n=len(retention),
            retention8_deterministic_successes=sum(qualified(r) for r in retention_det), retention8_deterministic_n=len(retention_det),
            cold_deterministic_successes=sum(qualified(r) for r in cold_det), cold_deterministic_n=1,
            cold_stochastic_successes=sum(qualified(r) for r in cold_stoch), cold_stochastic_n=len(cold_stoch)))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(15, 9.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=.53, wspace=.32)
    online_ax = fig.add_subplot(grid[0, :])
    phase_ax = fig.add_subplot(grid[1, 0])
    retention_ax = fig.add_subplot(grid[1, 1])
    cold_ax = fig.add_subplot(grid[1, 2])
    blue, teal, amber, slate = '#3264A5', '#007F76', '#AC6917', '#5D6170'

    for axis in (online_ax, phase_ax, retention_ax, cold_ax):
        axis.set_ylim(-.08, 1.19)
        axis.set_yticks([0, .25, .5, .75, 1.])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis='y', color='#E5E7EB', linewidth=.8)
        axis.set_axisbelow(True)
    positions = np.arange(len(bins))
    bars = online_ax.bar(positions, [b['successes'] / b['n'] for b in bins], color=blue, width=.73)
    for bar, b in zip(bars, bins):
        if b['partial_bin']:
            bar.set_hatch('///')
            bar.set_edgecolor('#163855')
        online_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .035,
                       f"{b['successes']}/{b['n']}" + ('\npartial bin' if b['partial_bin'] else ''),
                       ha='center', va='bottom', fontsize=9)
    online_ax.set_xticks(positions, [f"{b['first_episode']}–{b['last_episode']}" for b in bins])
    online_ax.set_title('A  Online training: completed episodes, changing policy', loc='left', pad=12)
    online_ax.set_xlabel('Online episode index · nonoverlapping groups of 100 · warm-up episodes excluded')
    online_ax.set_ylabel('Recorded qualified successes')
    if summary['unfinished_last_suffix']:
        online_ax.text(1, -.29, 'Final unfinished suffix excluded; its genuine nonterminal evidence is retained.',
                       transform=online_ax.transAxes, ha='right', fontsize=9, color=slate)

    names = ['Baseline\nbefore pretraining', 'SAC 500', 'SAC 1000', 'SAC 2000']
    x = np.arange(4)
    phase_values = [r['phase7_stochastic_successes'] / 5 for r in evaluations]
    phase_ax.plot(x, phase_values, color=teal, marker='o', linewidth=2, markersize=7)
    for pos, val, row in zip(x, phase_values, evaluations):
        phase_ax.annotate(f"{row['phase7_stochastic_successes']}/5", (pos, val), xytext=(0, 10), textcoords='offset points', ha='center')
    phase_ax.set_xticks(x, names, fontsize=8.5)
    phase_ax.set_title('B  Fixed checkpoints: prefix 7', loc='left', pad=12)
    phase_ax.set_xlabel('Same five stochastic seeds at each checkpoint')
    phase_ax.set_ylabel('Recorded qualified successes')

    retention_rows = [evaluations[0], evaluations[-1]]
    values = [r['retention8_stochastic_successes'] / 5 for r in retention_rows]
    retention_ax.bar([0, 1], values, color=amber, width=.48)
    for pos, val, row in zip([0, 1], values, retention_rows):
        retention_ax.text(pos, val + .04, f"{row['retention8_stochastic_successes']}/5", ha='center')
        retention_ax.text(pos, -.18, f"Deterministic: {row['retention8_deterministic_successes']}/1", ha='center', fontsize=9)
    retention_ax.set_xticks([0, 1], ['Baseline', 'SAC 2000'])
    retention_ax.set_xlim(-.6, 1.6)
    retention_ax.set_title('C  Fixed checkpoints: prefix 8', loc='left', pad=12)
    retention_ax.set_xlabel('Retention check · five separate stochastic seeds', labelpad=31)

    det = [r['cold_deterministic_successes'] for r in evaluations]
    cold_ax.plot(x, det, marker='s', color=slate, linestyle='-', label='Deterministic (1 attempt)')
    for pos, val in zip(x, det):
        cold_ax.annotate(f'{val}/1', (pos, val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    cold_value = evaluations[-1]['cold_stochastic_successes'] / 5
    cold_ax.plot([3.2], [cold_value], marker='D', color=teal, linestyle='none', markersize=7,
                 label='Stochastic (5 attempts, final only)')
    cold_ax.annotate(f"{evaluations[-1]['cold_stochastic_successes']}/5", (3.2, cold_value),
                     xytext=(10, -28 if cold_value > .7 else 30), textcoords='offset points', ha='left', fontsize=9)
    cold_ax.set_xticks(x, ['Baseline', '500', '1000', '2000'])
    cold_ax.set_xlim(-.35, 3.75)
    cold_ax.set_title('D  Cold reset: policy only', loc='left', pad=12)
    cold_ax.set_xlabel('Online SAC update count')
    cold_ax.legend(loc='upper left', bbox_to_anchor=(-.02, -.23), frameon=False, fontsize=8.5)

    fig.suptitle('Suffix SAC diagnostic: assisted training and separate cold-reset evaluations',
                 x=.06, y=.978, ha='left', fontsize=16, fontweight='bold')
    fig.text(.06, .94, 'One fixed prefix-7 stage · 256 frozen-actor warm-up episodes · 500 critic pretraining updates · 2000 online interactions',
             ha='left', fontsize=10.5, color=slate)
    fig.text(.06, .027,
             'Ratios describe these recorded trials, not robustness estimates. Training episodes can span changing policies; evaluation uses frozen checkpoints.\n'
             'Prefix 7 and prefix 8 execute recorded assistance first. Cold reset executes the policy alone. No new inference, simulation or training was run for this plot.',
             ha='left', va='bottom', fontsize=9, color=slate)
    fig.subplots_adjust(left=.065, right=.97, top=.87, bottom=.16)
    fig.savefig(D / 'outcomes.png', dpi=180)
    fig.savefig(D / 'outcomes.pdf')
    plt.close(fig)

    data = dict(kind='saved-results-visualization', source=str(SOURCE), source_final_manifest_sha256=sha(SOURCE / 'manifest.json'),
                source_summary_sha256=sha(SOURCE / 'summary.json'), online_completed_episode_bins=bins,
                scheduled_evaluations=evaluations, unfinished_last_suffix_excluded=summary['unfinished_last_suffix'],
                warmup_episodes_excluded_from_online_bins=summary['counts']['warmup_complete_episodes'],
                physical_or_learning_executions=0, policy_inference_calls=0,
                interpretation='Recorded outcomes only; changing-policy training is distinct from fixed-checkpoint evaluation; not a robustness-rate estimate',
                input_report_hashes=read_set, matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                training_runtime=runtime)
    write(D / 'plotted_data.json', data)
    write(D / 'manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(dict(plot=str(D / 'outcomes.png'), online_bins=bins, evaluations=evaluations), indent=2))


if __name__ == '__main__':
    main()
