# sanity check for the solution:
# using both the comment in the instance file and the solution, detect
# which semi-prime was factored, then also check if that semiprime was
# one that occurred in the original list of semiprimes

def bits2int(bits):
    res = 0
    pos = 1
    for bit in bits:
        res += pos * int(bit)
        pos *= 2
    return res

with open('../../semiprimes_100.txt') as fsp:
    semiprimes = fsp.readlines()

for i in range(4, 101):
    instance = '../../instances/multi_long/{:03}.dimacs'.format(i)
    solution = 'solved/{:03}_solution.dimacs'.format(i)
    try: 
        with open(instance) as finst:
            for line in finst:
                if line.startswith('c p: [T,'):
                    p_vars = map(int, line[8:-2].split(','))
                elif line.startswith('c q: [T,'):
                    q_vars = map(int, line[8:-2].split(','))
                    break # q is always the last line we need
    except:
        print('file not found: {}'.format(instance))
        continue
    p_bits = [True]
    q_bits = [True]
    try:
        with open(solution) as fsol:
            for line in fsol:
                if line.startswith('v '):
                    solution = map(int, line[2:-2].split())
    except:
        print('file not found: {}'.format(solution))
        continue
    for (var, sol) in zip(p_vars, solution):
        assert abs(sol) == var, "p_vars not in order"
        p_bits.append(sol > 0)
    for (var, sol) in zip(q_vars, solution):
        assert abs(sol) == var, "q_vars not in order"
        q_bits.append(sol > 0)
    p = bits2int(p_bits)
    q = bits2int(q_bits)
    if p > q:
        p, q = q, p
    sp_line = '{} {} {} {}\n'.format(i, p, q, p*q)
    found = sp_line in semiprimes
    print('{:03}: {} * {} = {} ({})'.format(i, p, q, p*q, found))
    
    

