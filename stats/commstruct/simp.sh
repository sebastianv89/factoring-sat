for i in ../../instances/long/*.dimacs; do
    f=$(basename "$i")
    ./maplecomsps -dimacs="long/$f" "$i"
done

for i in ../../instances/karatsuba/*.dimacs; do
    f=$(basename "$i")
    ./maplecomsps -dimacs="karatsuba/$f" "$i"
done
