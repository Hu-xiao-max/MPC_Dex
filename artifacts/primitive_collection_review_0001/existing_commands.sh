# Review-only command template: not executed. Use a future independently successful
# 40mm SAC checkpoint whose authenticated payload has online_updates > 0.
# Replace SAC40_CHECKPOINT and SAC40_CONFIG with immutable actual paths.
uv run --frozen python scripts/audit_policy.py \
  --checkpoint SAC40_CHECKPOINT --config SAC40_CONFIG \
  --output artifacts/frozen_sac40_force_audit_0001 \
  --episodes 5 --seed 1000000 --device cuda --torch-threads 1

# The copied checkpoint above is immutable even if the original training run advances.
# This generic command currently mislabels BC as SAC; use only diagnostically until fixed.
# It hardcodes Torch threads4, so do not claim bit-identical execution to a1thread audit.
uv run --frozen python -m mpc_dex evaluate \
  --checkpoint artifacts/frozen_sac40_force_audit_0001/checkpoint.pt \
  --config SAC40_CONFIG --output artifacts/frozen_sac40_joint_evaluation_0001 \
  --episodes 5 --seed 1000000 --device cuda

# Existing extraction CLI, only after a complete genuinely SAC-derived collection
# and the future provenance gate; current generic evaluation output is insufficient.
uv run --frozen python -m mpc_dex extract-primitive \
  --trajectory VERIFIED_FROZEN_SAC40_TRAJECTORIES.npz \
  --output artifacts/rotation_primitive.npz
