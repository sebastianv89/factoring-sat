count.txt: ../count.py
	python3 $< <../../semiprimes.txt >$@

plot_count.png: plot_count.py count.txt
	python3 $< <count.txt >$@
