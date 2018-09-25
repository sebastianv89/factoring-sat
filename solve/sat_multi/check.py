# sanity check for the solution:
# using both the comment in the instance file and the solution, detect
# which semi-prime was factored, then also check if that semiprime was
# one that occurred in the original list of semiprimes

from collections import Counter

def bits2int(bits):
    res = 0
    pos = 1
    for bit in bits:
        res += pos * int(bit)
        pos *= 2
    return res

with open('../../semiprimes_100.txt') as fsp:
    semiprimes = fsp.readlines()

for n in range(4, 101):
    # extract which bits encode p and q from the instance file
    instance = '../../instances/multi_long/{:03}.dimacs'.format(n)
    try: 
        with open(instance) as finst:
            for line in finst:
                if line.startswith('c p: [T,'):
                    p_vars = [int(v) for v in line[8:-2].split(',')]
                elif line.startswith('c q: [T,'):
                    q_vars = [int(v) for v in line[8:-2].split(',')]
                    break # q is always the last line we need
    except:
        print('file not found: {}'.format(instance))
        continue
    
    factored = Counter()
    for seed in range(1, 101):
        # recover the found solution for p and q from the solution file
        p_bits = [True]
        q_bits = [True]
        solution_file = 'solved/{:03}_{:03}_solution.dimacs'.format(n, seed)
        solution = None
        try:
            with open(solution_file) as fsol:
                for line in fsol:
                    if line.startswith('v '):
                        solution = map(int, line[2:-2].split())
        except:
            #print('file not found: {}'.format(solution_file))
            continue
        if solution is None:
            #print('{} contains no solution'.format(solution_file))
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
        sp_line = '{} {} {} {}\n'.format(n, p, q, p*q)
        found = sp_line in semiprimes
        if found:
            factored[(p,q)] += 1
        else:
            print('invalid solution: {:03} ({}): {} * {} = {} ({})'.format(n, seed, p, q, p*q, found))
    for ((p,q), count) in factored.most_common(5):
        print('{:2} bits: {:x} * {:x} = {:x} found {} times'.format(n, p, q, p*q, count))
    
