#!/bin/bash
# Tolerance grids of the sparse Case II (2026-11-16): m_k = 27 (s unknown) with eps_k = eps0 sqrt(kappa |I_k|),
# m_k = 3 (s known) with eps_k = eps0 sqrt(s |I_k|). Existing outputs are skipped.
cd "$(dirname "$0")" || exit 1
K="1 2 3 4 5 6 7 8 9 10"
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m3_epsgrid.json --methods dense sparse --m 3 --k $K --workers 9
python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m27_epsgrid.json --methods dense sparse --m 27 --k $K --workers 9
# q = 2 selects the smallest eta0 of the grids above, so eta0 is extended twice downwards
for g in q2ext q2ext2; do
  python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m27_epsgrid_$g.json --methods dense --m 27 --k $K --workers 9
  python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m3_epsgrid_$g.json --methods dense --m 3 --k $K --workers 9
done
python3 summarize_sparse_r2.py --tag sparse_d729
python3 plot_qsens_sparse_r2.py
