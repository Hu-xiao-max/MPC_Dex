"""Read-only verification of saved policy actions, densities and RNG advancement."""
from pathlib import Path
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
agent = SACAgent.load(D / 'checkpoint.pt', 'cuda')
summary = json.loads((D / 'summary.json').read_text())
rows = []
with torch.no_grad():
    for stage in summary['stages']:
        for report in stage['results']:
            name = 'deterministic' if report['deterministic'] else f"stochastic_{report['seed']}"
            dest = D / f"prefix_{stage['prefix_controls']:02d}" / name
            saved_rng = torch.load(dest / 'rng_initial.pt', weights_only=False)
            torch.set_rng_state(saved_rng['cpu'])
            torch.cuda.set_rng_state_all(saved_rng['cuda'])
            with np.load(dest / 'transitions.npz') as z:
                arrays = {key: z[key] for key in z.files}
            for index, ob in enumerate(arrays['obs']):
                tensor = torch.as_tensor(ob[None], device='cuda', dtype=torch.float32)
                mean, logstd = agent.actor(tensor)
                np.testing.assert_array_equal(mean[0].cpu().numpy(), arrays['raw_mean'][index])
                np.testing.assert_array_equal(logstd[0].exp().cpu().numpy(), arrays['latent_std'][index])
                if not arrays['policy_controlled'][index]:
                    assert np.isnan(arrays['log_prob'][index])
                    continue
                if report['deterministic']:
                    action, logp = mean.tanh(), agent.actor._log_prob(mean, logstd, mean)
                else:
                    action, logp = agent.actor.sample(tensor)
                np.testing.assert_array_equal(action[0].cpu().numpy(), arrays['actions'][index])
                assert float(logp[0]) == arrays['log_prob'][index]
            final_rng = torch.load(dest / 'rng_final.pt', weights_only=False)
            assert torch.equal(torch.get_rng_state(), final_rng['cpu'])
            assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), final_rng['cuda'], strict=True))
            rows.append({'episode': str(dest.relative_to(D)), 'actions_raw_means_stds_log_probs_final_rng_bitexact': True, 'actual_controls': len(arrays['actions'])})
result = {'status': 'complete', 'rows': rows, 'new_physics_controls': 0, 'new_training_updates': 0,
          'checkpoint_sha256': hashlib.sha256((D / 'checkpoint.pt').read_bytes()).hexdigest()}
(D / 'sampling_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
