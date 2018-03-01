semiprimes.txt: semiprimes.sage
	sage $< >$@

semiprimes_eqlen.txt: eqlen.py | semiprimes.txt
	python3 $< <semiprimes.txt >$@

clean:
	-rm -f semiprimes.txt semiprimes.sage.py
.PHONY: clean
