#!/bin/zsh
# Full pipeline of Section 5.2 (scenario wtp16). Run from NYCTaxi/ after the data are built (README.md).
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY run.py prepare                              # simulator fit, writes data/sim/
$PY -W ignore formal.py select wtp16            # validation Monday 2026-09-07
$PY -W ignore formal.py test wtp16              # timed runs on the three test Mondays, one after another
$PY -W ignore mosek_raw.py                      # AMLE-MOSEK runs with MOSEK's own solutions
$PY formal.py evaluate                          # replay of every saved run, writes formal.csv
$PY formal_report.py                            # formal_report.txt
$PY figure_paper.py figures                     # Figure 5
