#!/bin/bash
# One DAgger round: train the learner on every collection so far (first-order tube, second-order tube, DAgger rounds 1..R),
# evaluate it, then branch from its own rollouts along the spine into dagger_round$((R+1)).
# Usage: dagger_round.sh <R> [hidden] [steps] [base_collections] [floor_scale] [holdout_direction] [dagger_prefix]
set -e
R=$1; H=${2:-1024,1024,1024}; STEPS=${3:-100000}; BASE=${4:-tracking_tp}; FLOOR=${5:-0.005}; HOLD=${6:-5}; DP=${7:-dagger_tp_round}; HOR=${8:-1.5}; AVG=${9:-1}; AVGS=${10:-0}; TW=${11:-10000}; EXTRA=${LEARNER_EXTRA:-}
PY=/home/autel001/code/mpc_dex/.venv/bin/python; A=/home/autel001/code/mpc_dex/artifacts; T=$A/mpc_spine_tube_0001; F=$A/feedback_bc_0001
COLL=$BASE; for i in $(seq 1 $R); do COLL=$COLL,${DP}$i; done
TAG=${DP}_r${R}_h$(echo $H | tr ',' 'x')
echo "[round $R] training $TAG on $COLL"; $PY $F/train.py --tag $TAG --hidden $H --steps $STEPS --collections $COLL --holdout-direction $HOLD $EXTRA > $F/runs/$TAG.log 2>&1
grep -v "^step" $F/runs/$TAG.log | cut -c1-400
AG=$F/runs/$TAG/agent.pt; N=$((R+1)); echo "[round $R] branching from $TAG into ${DP}$N (floor $FLOOR)"
for rng in "0 2" "3 5" "6 9"; do set -- $rng; (nohup $PY $T/dagger_tube.py $AG $1 $2 1,2,3,4 $T/${DP}$N --floor-scale $FLOOR --horizon $HOR --avg-seeds $AVG --avg-steps $AVGS --track-weight $TW > $T/${DP}${N}_k${1}_${2}.log 2>&1 &); done
sleep 2; echo "[round $R] launched $(pgrep -fc 'python /home/autel001/code/mpc_dex/artifacts/mpc_spine_tube_0001/dagger_tube.py') dagger processes"
