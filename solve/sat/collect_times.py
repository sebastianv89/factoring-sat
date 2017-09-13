from collections import defaultdict
import sys

ws = defaultdict(list)
for line in sys.stdin:
    if line.startswith('#'):
        continue
    m, p, q, pq, w, u = line.split()
    m, p, q, pq = map(int, (m, p, q, pq))
    ws[(m, p, q, pq)].append(w)

print('#bitsize(pq) p q pq')
for (m, p, q, pq), w in sorted(ws.items()):
    print('{} {} {} {} {}'.format(m, p, q, pq, ' '.join(w)))
