solve: solve.py
	python3 $< <../../semiprimes.txt
.PHONY: solve

plot: plot.py
	python3 $< <timing.txt
.PHONY: plot

solutions.7z:
	7z a $@ solved/*.dimacs
