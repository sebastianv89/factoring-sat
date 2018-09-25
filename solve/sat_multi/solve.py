#!/usr/bin/env python3.6
# Solve all instances in the multi-sat dir
# TODO: upon killing with ctrl-c the subprocess fout file is already created but
#       empty and must be manually removed

import sys, os, glob, subprocess, datetime, time

for instance in sorted(glob.iglob('../../instances/multi_long/*.dimacs')):
    n = int(os.path.splitext(os.path.basename(instance))[0])
    call = ['./maplecomsps', '-rnd-init']
    for seed in range(1, 101):
        solution = './solved/{:03}_{:03}_solution.dimacs'.format(n, seed)
        if os.path.exists(solution):
            print('skipping already solved {}'.format(solution), file=sys.stderr)
            continue
        call.append('-rnd-seed={}'.format(seed))
        print('{}: {} ({})'.format(datetime.datetime.now(), instance, seed))
        with open(instance) as fin, open(solution, 'w', buffering=1) as fout:
            t0 = time.perf_counter()
            subprocess.run(call, stdin=fin, stdout=fout)
            t1 = time.perf_counter()
        with open('./timing.txt', 'a') as ftime:
            ftime.write('{} {} {}\n'.format(n, seed, t1-t0))
