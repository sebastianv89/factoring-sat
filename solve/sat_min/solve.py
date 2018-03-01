#!/usr/bin/env python3.6
# Solve the instances specified on each line of stdin
#  - {n} {p} {q} {pq}
# store only the shortest time

import sys, os, os.path, glob, math, subprocess, time, datetime

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    if n < 30:
        print('skipping small instances ({})'.format(n), file=sys.stderr)
        continue
    solved_file_templ = './solved/{:03}_{}_{}_solution_{}.dimacs'.format(n, p, q, '{}');
    if any(True for _ in glob.iglob(solved_file_templ.format('*'))):
        print('skipping solved {}'.format(solved_file_templ), file=sys.stderr)
        continue
    instance_file = '../../instances/long/{:03}_{}_{}.dimacs'.format(n, p, q)
    if not os.path.exists(instance_file):
        print('skipping non-existing {}'.format(instance), file=sys.stderr)
        continue
    with open(instance_file, 'rb') as fin:
        instance = fin.read()
    call_templ = './maplecomsps -rnd-init -rnd-seed={}'
    time_bound = 2**-18.1 * 2**(0.498*n) # estimated minimum solve time
    best_seed, solver_output = None, None
    for _ in range(10):
        for seed in range(1, 101): # testing 100 different seeds
            print('{}: {} seed={} ({}s)'.format(datetime.datetime.now(), instance_file, seed, time_bound))
            call = call_templ.format(seed, instance_file).split()
            time_bound_seconds = math.ceil(time_bound)
            try:
                t0 = time.perf_counter()
                cp = subprocess.run(call, input=instance, stdout=subprocess.PIPE, timeout=time_bound_seconds)
                t1 = time.perf_counter()
                if t1 - t0 < time_bound:
                    time_bound = t1 - t0
                    best_seed, solver_output = seed, cp.stdout
                    print('seed {}: {}s'.format(seed, time_bound))
            except subprocess.TimeoutExpired as e:
                print('seed {} timed out'.format(seed))
        if solver_output is not None:
            break
        time_bound *= 2
    if solver_output is None:
        print('No solution found for {}'.format(instance_file))
        with open('./unsolved.txt', 'a') as fout:
            fout.write('{} {} {} {} {}\n'.format(n, p, q, pq, time_bound));
    else:
        solved_file = solved_file_templ.format(best_seed)
        with open(solved_file, 'wb') as fout:
            fout.write(solver_output)
        with open('./timing.txt', 'a') as ftime:
            ftime.write('{} {} {} {} {} {}\n'.format(n, p, q, pq, best_seed, time_bound))
