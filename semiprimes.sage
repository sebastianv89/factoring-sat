from collections import defaultdict
from itertools import *

def bits(n):
    return ceil(log(n+1, 2))

# let m = bitsize(n). forall m <= M[i]: generate N[i] primes
N = [100, 10]
M = [50, 100]

# generate all semi-primes when there's less than N[0] per m
ns = defaultdict(set)
for n in count(2):
    pq = factor(n)
    if len(pq) == 2 and pq[0][1] == 1 and pq[1][1] == 1:
        p, q = pq[0][0], pq[1][0]
        if bits(q) - bits(p) > 1:
            continue
    elif len(pq) == 1 and pq[0][1] == 2:
        p, q = pq[0][0], pq[0][0]
    else:
        continue
    ns[bits(n)].add((p,q))
    if len(ns[bits(n)]) > N[0]:
        del ns[bits(n)]
        m = bits(n)
        break

# generate N[i] semi-primes per m (bitsize <= M[i])
for i in range(len(M)):
    while m <= M[i]:
        while len(ns[m]) < N[i]:
            p = random_prime(2^(m//2+1)-1, True, 2^(m//2-1))
            q = random_prime(2^(m//2+1)-1, True, 2^(m//2-1))
            if p > q:
                p, q = q, p
            if bits(q) - bits(p) > 1 or bits(p*q) != m:
                continue
            ns[m].add((p,q))
        m += 1

# output data
print('# bitsize(pq) p q pq')
for m, n in ns.iteritems():
    for p, q in sorted(n):
        print('{} {} {} {}'.format(m, p, q, p*q))
