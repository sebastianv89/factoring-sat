for d in */*; do
    if [ -d "$d" ]; then
        (cd "$d"; cat ./*/timing.txt | python3 ../../collect_times.py >timing.txt)
    fi
done
