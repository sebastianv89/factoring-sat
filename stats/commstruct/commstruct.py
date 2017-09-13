import sys
from igraph import *

encoding = 'long'
if len(sys.argv) > 1:
    encoding = sys.argv[1]

def file2graph(fname):
    edges = []
    with open(fname) as f:
        for line in f:
            if line.startswith('c'):
                continue
            elif line.startswith('p'):
                words = line.split()
                vertices = int(words[2])
            else:
                literals = map(int, line.split()[:-1])
                variables = list(map(abs, literals))
                for (i, v) in enumerate(variables):
                    for w in variables[(i+1):]:
                        edges.append((v-1, w-1))
    if vertices == 0:
        raise Exception('empty .dimacs file')
    g = Graph(vertices, edges)
    g.simplify()
    return g

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    fname = '{}/{:03}_{}_{}.dimacs'.format(encoding, n, p, q)
    try:
        g = file2graph(fname)
    except Exception as e:
        print('skipping {}: {}'.format(fname, e), file=sys.stderr)
        continue
    vd = g.community_fastgreedy()
    mod = g.modularity(vd.as_clustering())
    print('{} {} {} {} {}'.format(n, p, q, pq, mod))
