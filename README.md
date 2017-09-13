## Factoring semi-primes with SAT solvers

This repository contains scripts and data complementary to the
[paper](#TODO).

Most of the scripts are pretty small and self-explanatory.  In
addition most directories contain Makefiles that should give a clear
indication of how the scripts should be called.  Generate
`semiprimes.txt` first, then generate the SAT-instances and then let
the solvers solve those instances.

The Karatsuba instances were generated with
[ToughSat](https://toughsat.appspot.com/).  Reproducing the instances
will overload their service, so I did not attach the code to do so.

Running the benchmarks can take a long time, so for reproducibility of
the statistics the instances and solutions have been included in `.7z`
files.

Lastly, the `stats` directory contains measurements for for computing
the community structure and other metrics of the SAT instances.
