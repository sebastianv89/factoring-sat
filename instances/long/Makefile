count.txt: ../count.py
	python3 $< <../../semiprimes.txt >$@

plot_count.png: plot_count.py count.txt
	python3 $< <count.txt >$@

check:
	mkdir -p check
	bash checkAll.sh
.PHONY: check
