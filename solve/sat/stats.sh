for s in 'median' 'std' 'min'; do
    for d in */*; do
        if [ -d "$d" ]; then
            (cd "$d"; python3 ../../stats.py $s <timing.txt >$s.txt)
        fi
    done
done
