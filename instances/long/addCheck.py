import sys

n = int(sys.argv[1])
p = int(sys.argv[2])
q = int(sys.argv[3])
fname = '{:03}_{}_{}.dimacs'.format(n, p, q)

def setValue(n, bits):
    i = 0
    if bits[0] == 'T':
        i = 1
        n //= 2
    while n > 0:
        if n%2 == 0:
            print('-{} 0'.format(bits[i]))
        else:
            print('{} 0'.format(bits[i]))
        i += 1
        n //= 2

with open(fname) as f:
    for line in f:
        if line.startswith('c p:'):
            bits = line[6:-2].split(',')
            setValue(p, bits)
            print(line, end='')
        elif line.startswith('c q:'):
            bits = line[6:-2].split(',')
            setValue(q, bits)
            print(line, end='')
        else:
            print(line, end='')
