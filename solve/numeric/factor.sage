# factor semi-primes using whatever SageMath believes is best

import sys

for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n = sage_eval(words[3])
    t0 = walltime()
    factor(n)
    t1 = walltime(t0)
    words.append(str(t1))
    print(' '.join(words))
