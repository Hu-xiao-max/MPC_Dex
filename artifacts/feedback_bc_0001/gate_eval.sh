#!/bin/bash
# Learnability gate on a tube collection: large-actor learner (held-out direction) + per-time linear law.
# Usage: gate_eval.sh <collection> <tag> [holdout_direction=5] [steps=100000]
set -e
C=$1; TAG=$2; HOLD=${3:-5}; STEPS=${4:-100000}; PY=/home/autel001/code/mpc_dex/.venv/bin/python; F=/home/autel001/code/mpc_dex/artifacts/feedback_bc_0001
$PY $F/train.py --tag $TAG --hidden 1024,1024,1024 --steps $STEPS --collections $C --holdout-direction $HOLD > $F/runs/$TAG.log 2>&1 &
$PY $F/linear_law.py $C $HOLD > $F/runs/linear_law_$TAG.log 2>&1 &
wait
echo "== learner $TAG"; grep -v "^step" $F/runs/$TAG.log | grep -v Warning | cut -c1-500
echo "== linear law"; grep -v Warning $F/runs/linear_law_$TAG.log | grep -E "^ *[0-9]+ \||closed loop|held-out branches|j=[0-9]+:" | cut -c1-200
