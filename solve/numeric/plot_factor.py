import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

times = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    m, p, q, pq, t = line.split()
    m = int(m)
    t = float(t)
    times[m].append(t)

times = list(sorted(times.items()))
xs = [x for x, y in times]
ys = [y for x, y in times]

plt.boxplot(ys, positions=xs)
ax = plt.gca()
ax.set_xticks([xs[0]] + list(range(10, 91, 10)) + [xs[-1]])
ax.set_xticklabels([xs[0]] + list(range(10, 91, 10)) + [xs[-1]])
plt.ylim([0, 0.025])
plt.xlabel('Semiprime length (bits)')
plt.ylabel('Time (seconds)')
plt.savefig(sys.stdout.buffer, bbox_inches='tight')
