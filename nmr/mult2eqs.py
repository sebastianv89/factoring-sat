#!/usr/bin/env python3

import sys
from collections import Counter

def int2bits(n):
    bits = []
    while n > 0:
        bits.append(n & 1)
        n >>= 1
    return bits

def bits2string(x, y):
    if x.type == Bit.Type.VALUE:
        if x.value == 0:
            return '0'
        if x.value == 1:
            return str(y)
        if y.type == Bit.Type.VALUE:
            return str(Bit(Bit.Type.VALUE, x.value * y.value))
        return str(x.value) + str(y)
    if y.type == Bit.Type.VALUE and y.value == 1:
        return str(x)
    return str(x) + '*' + str(y)
        
def eqs2string(eqs):
    s = []
    for eq in eqs:
        lhs = [bits2string(x, y) for x, y in eq[0]]
        rhs = [bits2string(x, y) for x, y in eq[1]]
        s.append('+'.join(lhs) + '==' + '+'.join(rhs))
    return ',\n'.join(s)
    #return '\n'.join(s)

def mult2eqs(N, n, m):
    variables = Counter()
    bits = Bits()
    zero, one = Bit(Bit.Type.VALUE, 0), Bit(Bit.Type.VALUE, 1)
    ps = [one] + [bits.next(Bit.Type.P) for _ in range(n-2)] + [one]
    qs = [one] + [bits.next(Bit.Type.Q) for _ in range(m-2)] + [one]
    eqs = [([],[(one, Bit(Bit.Type.VALUE, ob))]) for ob in int2bits(N)]
    for offset, q in enumerate(qs):
        for i, p in enumerate(ps):
            eqs[offset+i][0].append((p,q))
    for i in range(len(eqs)):
        c = 2
        j = i+1
        while c <= len(eqs[i][0]):
            carry_var = bits.next(Bit.Type.CARRY)
            carry_coef = Bit(Bit.Type.VALUE, c)
            eqs[i][1].append((carry_coef, carry_var))
            if j >= len(eqs):
                eqs.append(([],[(one, zero)]))
            eqs[j][0].append((one, carry_var))
            j += 1
            c *= 2
    
    return eqs2string(eqs), str(bits)

def main():
    if len(sys.argv) <= 1 or len(sys.argv) >= 5:
        print('Usage: {} N [n] [m]'.format(sys.argv[0]), file=sys.stderr)
        exit()
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])
    else:
        n = m = (1 + N.bit_length()) // 2
    if len(sys.argv) >= 4:
        m = int(sys.argv[3])
    else:
        m = n
    eq_str, var_str = mult2eqs(N, n, m)
    #print(eq_str+'\n\n'+var_str)
    print('binarySimplify[\nJoin[{'+eq_str+'}],\n{'+var_str+'}]')

if __name__ == '__main__':
    main()
