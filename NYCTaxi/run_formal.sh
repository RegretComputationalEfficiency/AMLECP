#!/bin/bash
cd "$(dirname "$0")"
PY=python3
for s in wtp16 wtp14 wtp18; do $PY -W ignore formal.py select $s || exit 1; done
for s in wtp16 wtp14 wtp18; do $PY -W ignore formal.py test $s || exit 1; done
$PY formal.py evaluate && $PY formal_report.py && echo PIPELINE DONE
