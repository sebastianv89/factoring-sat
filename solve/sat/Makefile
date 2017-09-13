solve: solve.py
	python3 $< <../../semiprimes.txt
.PHONY: solve

collect: collect_times.sh collect_times.py
	bash $<
.PHONY: collect

stats: stats.sh stats.py collect
	bash $< 
.PHONY: stats

plot: plot.sh plot.py stats
	bash $<
.PHONY: plot

solutions.7z:
	7z a $@ */*/*/*solution.dimacs
