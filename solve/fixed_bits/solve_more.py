#!/usr/bin/env python3

import glob, os, random, subprocess, time

TMP_DIR = './tmp'
INST_DIR = '../../instances/long'
TIMING_FILE = './timing_more.txt'
SHUFFLE_FILE = './shuffle.txt'
TIMEOUT = 20 # seconds
RANDOM_SEEDS = 2
SHUFFLES = 3
INSTANCES_PER_SIZE = 3
SIZES = range(220, 300, 5)

def solve(fin, name):
    success = False
    for seed in range(1, RANDOM_SEEDS+1):
        try:
            t0 = time.perf_counter()
            subprocess.run(['./maplecomsps', '-rnd-init', f'-rnd-seed={seed}', fin],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=TIMEOUT)
            t1 = time.perf_counter()
            t = t1 - t0
            success = True
        except subprocess.TimeoutExpired:
            t = f'>{TIMEOUT}'
        with open(TIMING_FILE, 'a') as timing:
            timing.write(f'{name} {t}\n')
    return success

def int2bits(n):
    bits = []
    while n > 0:
        bits.append(n % 2 == 1)
        n //= 2
    return bits

def literals(bits, variables):
    literals = []
    for (i, v) in enumerate(variables):
        if i < len(bits) and bits[i]:
            literals.append(v)
        else:
            literals.append(-v)
    return literals

def get_base_instances(size):
    files = glob.glob(f'{INST_DIR}/{size:03}_*.dimacs')
    selected = random.sample(files, INSTANCES_PER_SIZE)
    for fname in selected:
        fname_base = os.path.splitext(os.path.basename(fname))[0]
        p, q = (int(x) for x in fname_base.split('_')[1:])
        p_bits, q_bits = int2bits(p), int2bits(q)
        with open(fname) as f:
            for line in f:
                if line.startswith('c p:'):
                    p_vars = [int(var) for var in line[8:-2].split(',')]
                    p_lits = literals(p_bits[1:], p_vars)
                elif line.startswith('c q:'):
                    q_vars = [int(var) for var in line[8:-2].split(',')]
                    q_lits = literals(q_bits[1:], q_vars)
                    break
                else:
                    continue
        yield (p_lits + q_lits, fname)

def gen_shuffles(size, lit_count):
    shuffles = []
    shuffle = list(range(lit_count))
    with open(SHUFFLE_FILE, 'a') as f:
        for s in range(SHUFFLES):
            random.shuffle(shuffle)
            f.write(f'{size} {s} {shuffle}\n')
            shuffles.append(shuffle[:])
    return shuffles

def reveal_file(revealed, base, literals):
    with open(base) as fin, open(revealed, 'w') as fout:
        pre, m = next(fin).rsplit(' ', 1)
        fout.write(f'{pre} {int(m) + len(literals)}\n')
        for line in fin:
            fout.write(line)
        for lit in literals:
            fout.write(f'{lit} 0\n')

def reveal_solve(fname, s, lits):
    fin = os.path.basename(fname)
    for k in range(3*len(lits)//4, 0, -1):
        revealed = f'{TMP_DIR}/{s}_{k}_{fin}'
        reveal_file(revealed, fname, lits[:k])
        name = f'{fin} {s} {k}'
        if not solve(revealed, name):
            os.remove(revealed)
            break # Stop when none were solved in time for k revealed bits
        os.remove(revealed)

def main():
    for size in SIZES:
        instances = list(get_base_instances(size))
        lit_count = len(instances[0][0])
        shuffles = gen_shuffles(size, lit_count)
        for (lits, fname) in instances:
            for (s, shuffle) in enumerate(shuffles):
                ls = [lits[i] for i in shuffle]
                reveal_solve(fname, s, ls)

if __name__ == '__main__':
    main()
