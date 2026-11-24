#!/bin/zsh
# Robustness of Section 5.2 to the median WTP multiplier: AMLE-PGD and HPF for the medians 1.4 and 1.8.
# Run from NYCTaxi/ after the data are built and run.py prepare has written data/sim/ (README.md).
cd "$(dirname "$0")"
PY=${PY:-python3}
for sc in wtp14 wtp18; do
  $PY -W ignore robust_wtp.py select $sc || { echo FAILED select $sc; exit 1; }   # validation Monday 2026-09-07
  $PY -W ignore robust_wtp.py test $sc || { echo FAILED test $sc; exit 1; }       # three test Mondays x three seeds
done
$PY -W ignore robust_wtp.py report || { echo FAILED report; exit 1; }            # robust_wtp_report.txt
