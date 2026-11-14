#!/bin/bash
# Reruns of every FIA analysis on the FIADB 9.5 table fia_db (paper/FIA_REBUILD_DESIGN.md section 5).
# Scripts unchanged except that the FIA name comes from experiments/data.py (DS.FIA = fia_db); same splits and
# seeds as the original runs (100 splits; 200 for the SC-GeoCCP slack; carbon 20 maps x 100 audits). <= 4 processes.
#   bash experiments/rerun_fia_db.sh coverage|carbon|ood|stress|r2      (logs in results/fia_db/logs/)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/miniforge3/envs/favela/bin/python -u -B"
W=4; DB=fia_db; RES=results/fia_db
mkdir -p $RES/logs $RES/carbon_random20
run() { local tag=$1; shift; echo "[$(date '+%F %T')] start $tag: $*"; "$@" > $RES/logs/$tag.log 2>&1; echo "[$(date '+%F %T')] done  $tag"; }

case "$1" in
coverage)
  run main        $PY experiments/run_main.py 100 $W $DB
  run baselines   $PY experiments/run_baselines.py 100 $W $DB
  run scorefns    $PY experiments/run_scorefns.py 100 $W $DB
  run scg_grid    $PY experiments/run_scgeoccp.py 100 $W $DB
  run scg_l1fix   env L1_FIXED=0.95 $PY experiments/run_scgeoccp.py 100 $W $DB
  run scg_slack   $PY experiments/run_scg_slack.py 200 $DB
  run mondrian    $PY experiments/run_mondrian.py 100 $W $DB
  run mondrian_r  $PY experiments/run_mondrian_rules.py 100 $W $DB
  run decisions   $PY experiments/run_decisions.py 100 $W $DB ;;
carbon)
  run carbon      $PY experiments/run_carbon.py 20 100 $W
  run carbon_r20  env CARBON_SELECT=random CARBON_OUT=$RES/carbon_random20 $PY experiments/run_carbon.py 20 100 $W
  run carbon_r10  env CARBON_SELECT=random $PY experiments/run_carbon.py 10 100 $W      # reuses the first 10 random maps
  run domain      $PY experiments/run_carbon_domain.py 20 100 $W
  run domain_r20  env CARBON_OUT=$RES/carbon_random20 $PY experiments/run_carbon_domain.py 20 100 $W random
  run domain_r10  $PY experiments/run_carbon_domain.py 10 100 $W random
  run units       $PY experiments/run_carbon_units.py 20 100 $W ;;
ood)
  run ood         $PY experiments/run_ood.py $W loso:$DB:siteheight loso:$DB:full block:$DB:k25 block:$DB:k100 block:$DB:random ;;
stress)
  run st_twomap   $PY experiments/run_stress.py twomap 100 $W $DB
  run st_signal   $PY experiments/run_stress.py signal 100 $W $DB ;;
r2)
  run heldout_r2  $PY -I experiments/heldout_r2.py 10 $W ;;
esac
