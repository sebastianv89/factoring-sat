# plot variable/clause count

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
import sys

ms, vs, cs, cls = [], [], [], []
for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    m, p, q, pq, v, c = map(int, words[:6])
    cl = float(words[6])
    ms.append(m)
    vs.append(v)
    cs.append(c)
    cls.append(cl)

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

def func(x, a, b, c):
    return a*(x**math.log2(3)) + b*x + c

vfit, _ = curve_fit(func, ms, vs)
vrsquared = rsquared(vs, np.array([func(m, *vfit) for m in ms]))
print('vfit: {} (r^2: {})'.format(vfit, vrsquared), file=sys.stderr)

cfit, _ = curve_fit(func, ms, cs)
crsquared = rsquared(cs, np.array([func(m, *cfit) for m in ms]))
print('cfit: {} (r^2: {})'.format(cfit, crsquared), file=sys.stderr)

print('average clause length: {}'.format(np.mean(cls)), file=sys.stderr)

xs = np.linspace(1, ms[-1])
plt.plot(ms, vs, '.', label='Variables', color='r')
plt.plot(xs, func(xs, *vfit), color='r')
plt.plot(ms, cs, '.', label='Clauses', color='g')
plt.plot(xs, func(xs, *cfit), color='g')
plt.xlabel('Multiplication output size (bits)')
plt.legend(loc='best')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
