# Generate one instances per semiprime size

import sys, subprocess

semiprimes = []
last_n = None

def gensat():
    call = ['../gensat'] + list(map(str, semiprimes))
    instance = '{:03}.dimacs'.format(last_n)
    with open(instance, 'w') as fout:
        subprocess.run(call, stdout=fout)

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    if p == 2 or q == 2:
        continue # multi-output instances not implemented for even primes
    if last_n is None:
        last_n = n
        semiprimes = [pq]
    elif n == last_n:
        semiprimes.append(pq)
    else:
        gensat()
        # reset for next n
        last_n = n
        semiprimes = [pq]

gensat()
