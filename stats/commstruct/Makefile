long/commstruct.txt: commstruct.py
	python3 $< long <../../semiprimes.txt >$@

karatsuba/commstruct.txt: commstruct.py
	python3 $< karatsuba <../../semiprimes.txt >$@

long/comm_timing.txt: comm_timing.py | long/commstruct.txt
	python3 $< long >$@

karatsuba/comm_timing.txt: comm_timing.py | karatsuba/commstruct.txt
	python3 $< karatsuba >$@

long/comm_timing.eps: plot_commstruct.py | long/comm_timing.txt
	python3 $< long <long/comm_timing.txt >$@

karatsuba/comm_timing.eps: plot_commstruct.py | karatsuba/comm_timing.txt
	python3 $< karatsuba <karatsuba/comm_timing.txt >$@

simplify:
	bash simp.sh
.phony: simplify
