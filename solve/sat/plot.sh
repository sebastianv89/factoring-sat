for solv in 'cryptominisat5' 'maplecomsps'; do
    for enc in 'long' 'karatsuba'; do
        if [ -d "$solv/$enc" ]; then
            for stat in 'median' 'mean' 'min'; do
                (>&2 echo $stat)
                (cd "$solv/$enc"; python3 ../../plot.py "$solv" "$enc" "$stat" <$stat.txt >$stat.png)
            done
        fi
    done
done
