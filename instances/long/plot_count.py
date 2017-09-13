# plot variable/clause count

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sys

def rsquared(y, fit):
    y_mean = np.mean(y)
    ss_tot = np.sum((y-y_mean)**2)
    ss_res = np.sum((y-fit)**2)
    return 1.0 - ss_res/ss_tot

ns, vs, cs, cls = [], [], [], []
for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n, p, q, pq, v, c = map(int, words[:6])
    cl = float(words[6])
    ns.append(n)
    vs.append(v)
    cs.append(c)
    cls.append(cl)
    
vcoefs = poly.polyfit(ns, vs, 2)
vfit = poly.Polynomial(vcoefs)
vrsquared = rsquared(vs, vfit(ns))
print('vfit: {} (r^2: {})'.format(vfit, vrsquared), file=sys.stderr)

ccoefs = poly.polyfit(ns, cs, 2)
cfit = poly.Polynomial(ccoefs)
crsquared = rsquared(cs, cfit(ns))
print('cfit: {} (r^2: {})'.format(cfit, crsquared), file=sys.stderr)

print('average clause length: {}'.format(np.mean(cls)), file=sys.stderr)

xs = np.linspace(ns[0], ns[-1])
plt.plot(ms, vs, '.', label='Variables', color='r')
plt.plot(xs, vfit(xs), color='r')
plt.plot(ms, cs, '.', label='Clauses', color='g')
plt.plot(xs, cfit(xs), color='g')
plt.xlabel('Multiplication output size (bits)')
plt.legend(loc='best')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
