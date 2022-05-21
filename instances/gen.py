# Generate the instances specified on each line of stdin

import sys, subprocess

encoding = 'long'
if len(sys.argv) > 1:
    encoding = sys.argv[1]

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    instance = '{:03}_{}_{}.dimacs'.format(n, p, q)
    sp = command.format(pq).split()
    sp = f'cabal run gensat {pq}'.split()
    with open(instance, 'w') as fout:
        subprocess.call(sp, cwd='../gensat', stdout=fout)
