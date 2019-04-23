solve: ./solve.py
	python3.6 solve.py
.PHONY: solve

check: ./check.py
	python3.6 check.py
.PHONY: check

plot.pdf: ./plot.py timing.txt
	python3 $< <timing.txt >$@ 2>plot.out

solutions.7z:
	7z a $@ solved/*.dimacs
