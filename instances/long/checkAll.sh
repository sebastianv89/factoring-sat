#!/usr/bin/bash

for f in *.dimacs; do
    a=(${f//[._]/ })
    python3 addCheck.py ${a[0]} ${a[1]} ${a[2]} >check/$f
done
