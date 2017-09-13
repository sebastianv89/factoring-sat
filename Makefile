semiprimes.txt: semiprimes.sage
	sage $< >$@

clean:
	-rm -f semiprimes.txt semiprimes.sage.py
.PHONY: clean
