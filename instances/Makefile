karatsuba long: gensat gen.py
	(cd $@; python3 ../gen.py $@ <../../semiprimes.txt)

gensat: gensat.hs
	ghc -O -o $@ $<

instances.7z:
	7z a $@ */*.dimacs

clean:
	-rm -f gensat gensat.dyn_hi gensat.dyn_o gensat.hi gensat.o
.PHONY: clean
