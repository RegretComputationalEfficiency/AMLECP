#!/bin/bash
# Sparse Case II of Figure EC.1 and Appendix B: heavy-tailed features, s = 6 nonzero parameters, T = 50,000,
# 10 trials. kappa = 730 (d = 729) with m_k = 27 and 3, and kappa = 201 (d = 200) with m_k = 14 and 3.
# Jobs whose output exists are skipped. Then: python3 summarize_sparse_r2.py --tag <tag>; python3 plot_qsens_sparse_r2.py
set -e
cd "$(dirname "$0")"
K="1 2 3 4 5 6 7 8 9 10"
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m27.json --m 27 --k $K --workers 9
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m27_qsens.json --methods dense --m 27 --k $K --workers 9
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m3.json --m 3 --k $K --workers 9
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m3_qsens.json --methods dense --m 3 --k $K --workers 9
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m3_qsens_ext.json --methods dense --m 3 --k $K --workers 9
python3 sparse_r2.py --d 200 --T 50000 --tag sparse_d200 --grid-json grid_sparse_d200_m14.json --m 14 --k $K --workers 9
python3 sparse_r2.py --d 200 --T 50000 --tag sparse_d200 --grid-json grid_sparse_d200_m3.json --m 3 --k $K --workers 9
