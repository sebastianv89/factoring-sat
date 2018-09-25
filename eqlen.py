# filter semiprimes where p.bit_length() != q.bit_length()

import sys

for line in sys.stdin:
    if line.startswith('#'):
        print(line, end='')
        continue
    n, p, q, pq = map(int, line.split())
    if p == 2 or p.bit_length() != q.bit_length():
        print('#', line, end='')
    else:
        print(line, end='')
