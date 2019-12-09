from collections import defaultdict
from itertools import *

def bits(n):
    return ceil(log(n+1, 2))

m = 105
while m <= 300:
    count = 0
    while count < 5:
        p = random_prime(2^(m//2+1)-1, True, 2^(m//2-1))
        q = random_prime(2^(m//2+1)-1, True, 2^(m//2-1))
        if p > q:
            p, q = q, p
        if bits(q) - bits(p) > 1 or bits(p*q) != m:
            continue
        print('{} {} {} {}'.format(m, p, q, p*q))
        count += 1
    m += 5
