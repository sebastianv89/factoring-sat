solve: solve.py
	python3 $< <../../semiprimes.txt
.PHONY: solve

plot.png: plot.py
	python3 $< <timing.txt >plot.png 2>plot.out

solutions.7z:
	7z a $@ solved/*.dimacs
