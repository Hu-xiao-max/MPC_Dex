# Real frozen-SAC collector integration

The new collector executed exactly five deterministic attempts from archived bootstrap0005 checkpoint6, with BC12000, critic-pretraining0 and actual online SAC1745 updates. It used the unchanged40mm horizon100 contract, CUDA singleton inference, one intra/inter-op thread and TF32 disabled. The checkpoint and numerical sources were copied before execution.

All five attempts fail at control7 due to force-bearing palm support. Each ends at force-qualified rotation1.0032547032686967rad and peak palm normal force1.7862867488820415N. The collector returns2, preserves all five failed episodes, sets eligible_for_primitive=false, and emits no successful_joint_trajectories.npz.

Each attempt matches the previously independent final-checkpoint deterministic audit exactly: observations, actions, rewards, next observations, terminal flags, qpos, qvel, targets, full MuJoCo integration state, and all350physics samples have zero difference. Initial and terminal measured joint frames are retained. Seeds42–46 produce one unique initial state and one unique measured trajectory; these are five repeated executions, not reset-diversity or robustness evidence.

This verifies the real collector and its refusal to export a failed SAC policy. It does not establish a successful SAC primitive or run any new learning. compare_reference.py only reads saved arrays. comparison.json records source hashes and exact comparisons.
