#!/usr/bin/env python3

import os
import subprocess
import time

INST_DIR = '../../instances/fixed_bits'

def solve(timing, fin, name):
    for r in range(5):
        t0 = time.perf_counter()
        subprocess.run(['./maplecomsps', '-rnd-init', f'-rnd-seed={r+1}', fin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        timing.write(f'{name} {t1 - t0}\n')

def solve_size(size):
    subdir = f'{INST_DIR}/lsb'
    print('lsb')
    with open('timing_lsb.txt', 'a') as timing:
        for fin in sorted(os.listdir(subdir)):
            if int(fin[:3]) != size:
                continue
            solve(timing, f'{subdir}/{fin}', fin)

    subdir = f'{INST_DIR}/msb'
    print('msb')
    with open('timing_msb.txt', 'a') as timing:
        for fin in sorted(os.listdir(subdir)):
            if int(fin[:3]) != size:
                continue
            solve(timing, f'{subdir}/{fin}', fin)

    subdir = f'{INST_DIR}/random/q_only'
    print('random q')
    with open('timing_random_q.txt', 'a') as timing:
        for fin in sorted(os.listdir(subdir)):
            if int(fin[:3]) != size:
                continue
            solve(timing, f'{subdir}/{fin}', fin)

    subdir = f'{INST_DIR}/random/both'
    print('random both')
    with open('timing_random_both.txt', 'a') as timing:
        for fin in sorted(os.listdir(subdir)):
            if int(fin[:3]) != size:
                continue
            solve(timing, f'{subdir}/{fin}', fin)

def main():
    for size in range(20, 51):
        print(size)
        solve_size(size)

if __name__ == '__main__':
    main()
