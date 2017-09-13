import sys

encoding = 'long'
if len(sys.argv) > 1:
    encoding = sys.argv[1]
    
lines = {}
with open(encoding + '/commstruct.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        n, p, q, pq, mod = line.split()
        lines[pq] = line[:-1]

with open('../../solve/sat/maplecomsps/' + encoding + '/min.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        n, p, q, pq, t = line.split()
        if not pq in lines:
            print('Skipping {} (no commstruct)'.format(pq), file=sys.stderr)
            continue
        print(lines[pq], t)
        del lines[pq]

for pq, line in lines.items():
    print('No timing found for {}'.format(pq), file=sys.stderr)
