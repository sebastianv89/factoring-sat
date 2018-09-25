#!/bin/bash

set -f
while IFS= read -ra line; do
    [[ $line = \#* ]] && continue
    [ -z "$line" ] && continue
    a=( $line )
    n=${a[0]}
    pq=${a[3]}
    echo "generating eqs for ${n}_$pq"
    python3 mult2eqs.py "$pq" | wolfram -script nmr.wls >"simplified/${n}_$pq.txt"
done
