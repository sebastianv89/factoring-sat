semiprimes_factor.txt: factor.sage
	sage $< <../../semiprimes.txt >$@

semiprimes_trial_division.txt: trial_division.sage
	sage $< <../../semiprimes.txt >$@

semiprimes_factor.pdf: plot_factor.py semiprimes_factor.txt
	python3 $< <semiprimes_factor.txt >$@

semiprimes_trial_division.pdf: plot_trialdiv.py semiprimes_trial_division.txt
	python3 $< <semiprimes_trial_division.txt >$@

clean:
	-rm -f *.sage.py
.PHONY: clean
