import sys
import matplotlib.pyplot as plt

xs, ys = [], []
for line in sys.stdin:
    if line.startswith('#'):
        continue
    words = line.split()
    n = int(words[0])
    t = float(words[1])
    xs.append(n)
    ys.append(t)

plt.plot(xs, ys)
plt.yscale('log')
plt.show()
