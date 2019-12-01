solve_long: solve.py
	python3 $< long <../../semiprimes.txt
solve_karatsuba: solve.py
	python3 $< karatsuba <../../semiprimes.txt
.PHONY: solve_long solve_karatsuba

plot_long.eps: plot.py timing_long.txt
	python3 $< <timing_long.txt >$@ 2>plot_long.out

plot_karatsuba.eps: plot.py timing_karatsuba.txt
	python3 $< <timing_karatsuba.txt >$@ 2>plot_karatsuba.out

solutions.7z:
	7z a $@ solved_long/*.dimacs solved_karatsuba/*.dimacs
