#/usr/bin/env python3

from collections import defaultdict
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('classic')

# upper bound on the number of operations per second for the SAT solver
OP_SEC = 10**10

def quantum_speedup(t):
    '''Expected quantum speedup, given classical runtime t.
    Assumptions:
     - classical method achieved the upper bound on operations/sec
     - full computation benefits from quadratic speedup
     - quantum computer has same clock speed as the classical computer
    '''
    ops = t * OP_SEC
    qops = math.sqrt(ops)
    qt = qops / OP_SEC
    return qt

def plot():
    domain = range(2, 1000)
    xs = list(domain)

    # expected runtime
    def sat_runtime(n):
        return 2**(.495*n - 16.4)
    exp = list(map(sat_runtime, domain))
    plt.plot(xs, exp, 'r', label='classical solver (expected runtime)',)
    
    # expected runtime (quantum speedup)
    qexp = list(map(quantum_speedup, exp))
    plt.plot(xs, qexp, 'r--', label='quantum solver (expected runtime)')
    
    # 1/10000 expected runtime
    def sat_min_min(n):
        return 2**(.217*n - 14.7)
    pr_exp = list(map(sat_min_min, domain))
    plt.plot(xs, pr_exp, 'b', label='classical solver ("easy" semi-primes)')

    # 1/10000 expected runtime (quantum speedup)
    pr_qexp = list(map(quantum_speedup, pr_exp))
    plt.plot(xs, pr_qexp, 'b--', label='quantum solver ("easy" semi-primes)')

    # trial division (experimental)
    def trialdiv_runtime(n):
        return 2**(.496*n - 28.8)
    td = list(map(trialdiv_runtime, domain))
    plt.plot(xs, td, 'g', label='trial division')

    # Number Field Sieve
    def nfs_runtime(n):
        '''L_n[1/3, (64/9)^1/3 + o(1)]'''
        x = (64.0/9.0)**(1/3)
        ln = n / math.log2(math.e)
        lln = math.log(ln)
        ops = math.exp(x * ln**(1/3) * lln**(2/3))
        return ops / OP_SEC;
    nfs = list(map(nfs_runtime, domain))
    plt.plot(xs, nfs, 'k', label='number field sieve')

    plt.yscale('log')
    plt.xlabel('$n$: semiprime length (bits)')
    plt.ylabel('$T(n)$: time in seconds')
    #plt.ylim(top=10.0**18.0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('combined.png');

def main():
    plot()

if __name__ == '__main__':
    main()

