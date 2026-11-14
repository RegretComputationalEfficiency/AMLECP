#!/bin/bash
# AMLE-PGD runs of Section 5.1 and Appendix B (solver solvers_r2.solve_pgd, R2_PGD=apg): results/dense_alg1
# (Table 2, Figure 4), results/qsens_tol_alg1 (Figure EC.1(a)) and the methods dense and sparse of
# results/sparse_d729 (Table 3, Figure EC.1(b)-(c)). AMLE-MOSEK, ILQX, SPO+ and AdaGrad are run by the commands
# in README.md. Run with: caffeinate -i bash queue_apg.sh   (existing outputs are skipped)
cd "$(dirname "$0")" || exit 1
mkdir -p logs
export R2_PGD=apg
K="1 2 3 4 5 6 7 8 9 10"
python3 run_r2.py --design dense --tag dense_alg1 --snr low mid high --methods PGD --grid main \
    --grid-json grid_dense.json --workers 9 > logs/dense_alg1_pgd_rerun.log 2>&1
echo "dense_alg1 PGD: exit $? ok=$(grep -c ' ok ' logs/dense_alg1_pgd_rerun.log) error=$(grep -c ' error ' logs/dense_alg1_pgd_rerun.log)"
python3 run_r2.py --design dense --tag qsens_tol_alg1 --snr mid --methods PGD --grid main \
    --grid-json grid_qsens_tol.json --workers 9 > logs/qsens_tol_alg1_rerun.log 2>&1
echo "qsens_tol_alg1: exit $? ok=$(grep -c ' ok ' logs/qsens_tol_alg1_rerun.log) error=$(grep -c ' error ' logs/qsens_tol_alg1_rerun.log)"
: > logs/sparse_d729_pgd_rerun.log
for spec in "grid_sparse_d729_m27.json 27 dense sparse" "grid_sparse_d729_m27_qsens.json 27 dense" \
            "grid_sparse_d729_m3.json 3 dense sparse" "grid_sparse_d729_m3_qsens.json 3 dense" \
            "grid_sparse_d729_m3_qsens_ext.json 3 dense"; do
    set -- $spec
    g=$1; m=$2; shift 2
    python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json $g --methods "$@" --m $m --k $K \
        --workers 9 >> logs/sparse_d729_pgd_rerun.log 2>&1
done
echo "sparse_d729 PGD: ok=$(grep -c ' ok ' logs/sparse_d729_pgd_rerun.log) error=$(grep -c ' error ' logs/sparse_d729_pgd_rerun.log)"
