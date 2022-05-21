# plot variable/clause count

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sys

ns, alphas = [], []
for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n, _, _, _, v, c = map(int, words[:6])
    ns.append(n)
    alphas.append(c / v)
    
plt.plot(ns, alphas)
plt.xlabel('Multiplication output size (bits)')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
