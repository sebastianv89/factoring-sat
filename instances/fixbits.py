#!/usr/bin/env python3

import glob
import os
import random
import shutil

INPUT_DIR = './long'
OUTPUT_DIR = './fixed_bits'
SIZES = range(20, 51)
PER_SIZE = 5
RANDOM_SELECTIONS = 5

def get_instances(n, size):
    files = glob.glob(f'{INPUT_DIR}/{size:03}_*.dimacs')
    selected = random.sample(files, n)
    res = []
    for fname in selected:
        fname_base = os.path.splitext(os.path.basename(fname))[0]
        p, q = (int(x) for x in fname_base.split('_')[1:])
        with open(fname) as f:
            for line in f:
                if line.startswith('c p:'):
                    p_vars = [0] + [int(var) for var in line[8:-2].split(',')]
                elif line.startswith('c q:'):
                    q_vars = [0] + [int(var) for var in line[8:-2].split(',')]
                    break
                else:
                    continue
        res.append(((p, p_vars), (q, q_vars), fname))
    return res

def int2bits(n):
    bits = []
    while n > 0:
        bits.append(n % 2 == 1)
        n //= 2
    return bits

def make_lsbs(instance):
    ((p, p_vars), (q, q_vars), fin) = instance
    fin_base = os.path.splitext(os.path.basename(fin))[0]
    p_bits = int2bits(p)
    q_bits = int2bits(q)
    for i in range(1, len(q_vars)):
        for j in [0, i]:
            if j == len(p_vars):
                continue
            fout = f'{OUTPUT_DIR}/lsb/{fin_base}_p{j:02}_q{i:02}.dimacs'
            shutil.copyfile(fin, fout)
            with open(fout, 'a') as f:
                for pj in range(1, j+1):
                    if pj >= len(p_bits) or not p_bits[pj]:
                        f.write('-')
                    f.write(f'{p_vars[pj]} 0\n')
                for qi in range(1, i+1):
                    if qi >= len(q_bits) or not q_bits[qi]:
                        f.write('-')
                    f.write(f'{q_vars[qi]} 0\n')

def make_msbs(instance):
    ((p, p_vars), (q, q_vars), fin) = instance
    fin_base = os.path.splitext(os.path.basename(fin))[0]
    p_bits = int2bits(p)
    q_bits = int2bits(q)
    for i in range(1, len(q_vars)-1):
        for j in [0, i]:
            if j == len(p_vars):
                continue
            fout = f'{OUTPUT_DIR}/msb/{fin_base}_p{j:02}_q{i:02}.dimacs'
            shutil.copyfile(fin, fout)
            with open(fout, 'a') as f:
                for pj in range(len(p_vars)-1, len(p_vars)-1-j, -1):
                    if pj >= len(p_bits) or not p_bits[pj]:
                        f.write('-')
                    f.write(f'{p_vars[pj]} 0\n')
                for qi in range(len(q_vars)-1, len(q_vars)-1-i, -1):
                    if qi >= len(q_bits) or not q_bits[qi]:
                        f.write('-')
                    f.write(f'{q_vars[qi]} 0\n')

def make_random_qs(instance):
    ((p, p_vars), (q, q_vars), fin) = instance
    fin_base = os.path.splitext(os.path.basename(fin))[0]
    p_bits = int2bits(p)
    q_bits = int2bits(q)
    for r in range(RANDOM_SELECTIONS):
        q_shuffled = q_vars[1:]
        random.shuffle(q_shuffled)
        for i in range(1, len(q_shuffled)):
            fout = f'{OUTPUT_DIR}/random/q_only/{fin_base}_{i:02}_{r}.dimacs'
            shutil.copyfile(fin, fout)
            with open(fout, 'a') as f:
                for j in range(i):
                    qj = q_vars.index(q_shuffled[j])
                    if qj >= len(q_bits) or not q_bits[qj]:
                        f.write('-')
                    f.write(f'{q_vars[qj]} 0\n')

def make_random_both(instance):
    ((p, p_vars), (q, q_vars), fin) = instance
    fin_base = os.path.splitext(os.path.basename(fin))[0]
    p_bits = int2bits(p)
    q_bits = int2bits(q)
    for r in range(RANDOM_SELECTIONS):
        shuffled = p_vars[1:] + q_vars[1:]
        random.shuffle(shuffled)
        for i in range(1, len(shuffled)):
            fout = f'{OUTPUT_DIR}/random/both/{fin_base}_{i:03}_{r}.dimacs'
            shutil.copyfile(fin, fout)
            with open(fout, 'a') as f:
                for j in range(i):
                    try:
                        pj = p_vars.index(shuffled[j])
                        bit_var = p_vars[pj]
                        if pj >= len(p_bits) or not p_bits[pj]:
                            bit_var = -bit_var
                    except ValueError:
                        qj = q_vars.index(shuffled[j])
                        bit_var = q_vars[qj]
                        if qj >= len(q_bits) or not q_bits[qj]:
                            bit_var = -bit_var
                    f.write(f'{bit_var} 0\n')

def main():
    for size in SIZES:
        instances = get_instances(PER_SIZE, size)
        for instance in instances:
            make_lsbs(instance)
            make_msbs(instance)
            make_random_qs(instance)
            make_random_both(instance)

if __name__ == '__main__':
    main()
