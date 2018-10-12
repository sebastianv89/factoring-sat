## Factoring semi-primes with SAT solvers

This repository contains scripts and data complementary to the
[paper](#TODO insert link).

Most of the scripts are pretty small and self-explanatory.  In
addition most directories contain Makefiles that should give a clear
indication of how the scripts should be called.  Generate
`semiprimes.txt` first, then generate the SAT-instances and then let
the solvers solve those instances.  Make sure you include a (symbolic)
link to the solvers in the corresponding directories.

The `solve` directory has several subdirectories.  The `numeric`
directory measures the runtime of factoring with number-theoretical
methods.  The `sat` directory measures runtime of factoring with
SAT solvers.  The `sat_min` directory takes a shortcut when you want
to determine the minimum runtime: try many seeds and stop any solver
that runs longer than the currently minimum runtime for the current
instance.  The `sat_multi` directory collapses multiple instances
into one by generating a single multiplication circuit, then fanning
out the resulting semiprime, checking the output against many semi-
primes and combining the results with a massive or-gate.  Thus if
there is one easy number to factor, we may hope that the SAT solver
is able to find it and focus on factoring that number.  (Although
from the results we conclude it is not able to find such a number.)
The `zoom` directory inspects some interesting distributions.

The Karatsuba instances were generated with
[ToughSat](https://toughsat.appspot.com/).  Reproducing the instances
will overload their service, so I did not attach the code to do so.

Running the benchmarks can take a long time, so for reproducibility of
the statistics the instances and solutions have been included in `.7z`
files.

Lastly, the `stats` directory contains measurements for for computing
the community structure and other metrics of the SAT instances.
