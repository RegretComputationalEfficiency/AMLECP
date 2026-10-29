#!/bin/bash
# Rerun of the q-sensitivity study (Figure EC.1) with the Algorithm 1 exploration schedule (env_r2.schedule),
# run with:  caffeinate -i bash queue_qsens_alg1.sh [PID]
# If a PID is given (the dense_alg1 rerun), wait for it to finish first. Only the runs used in Figure EC.1 are
# repeated (see plot_qsens_r2.py):
#   Case I : AMLE-MOSEK for q = 2 and 1 + log kappa (grid_qsens.json) and AMLE-PGD at the common tolerance
#            (grid_qsens_tol.json); q = 4 comes from results/dense_alg1;
#   Case II: all q, design heavy100 (grid_qsens_heavy100.json).
# Interrupted runs resume with the same command (existing outputs are skipped).
cd "$(dirname "$0")" || exit 1
mkdir -p logs
if [ -n "$1" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 120; done
fi
python3 run_r2.py --design dense --tag qsens_alg1 --snr mid --methods MOSEK --grid main \
    --grid-json grid_qsens.json --workers 9 > logs/qsens_alg1.log 2>&1
echo "qsens_alg1: exit $? ok=$(grep -c ' ok ' logs/qsens_alg1.log) error=$(grep -c ' error ' logs/qsens_alg1.log)"
python3 run_r2.py --design dense --tag qsens_tol_alg1 --snr mid --methods PGD --grid main \
    --grid-json grid_qsens_tol.json --workers 9 > logs/qsens_tol_alg1.log 2>&1
echo "qsens_tol_alg1: exit $? ok=$(grep -c ' ok ' logs/qsens_tol_alg1.log) error=$(grep -c ' error ' logs/qsens_tol_alg1.log)"
python3 run_r2.py --design heavy100 --tag qsens_heavy100_alg1 --snr mid --methods MOSEK PGD --grid main \
    --grid-json grid_qsens_heavy100.json --workers 9 > logs/qsens_heavy100_alg1.log 2>&1
echo "qsens_heavy100_alg1: exit $? ok=$(grep -c ' ok ' logs/qsens_heavy100_alg1.log) error=$(grep -c ' error ' logs/qsens_heavy100_alg1.log)"
