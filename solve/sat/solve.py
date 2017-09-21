# Solve the instances specified on each line of stdin
#  - {n} {p} {q} {pq}
# Assumes:
#  - cwd structure: ./{solver}/{encoding}/{seed}/
#  - solver executable: ./{solver}/{solver}
#  - instances: ./{idir}/{encoding}/{n:03}_{p}_{q}.dimacs

import sys, os, os.path, subprocess, time, datetime, resource

idir = '../../instances'
if len(sys.argv) > 1:
    idir = sys.argv[1]

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    for solver in os.listdir('.'):
        if not os.path.isdir(solver):
            continue
        ex = os.path.join(solver, solver)
        if solver == 'cryptominisat5':
            call = '{ex} --random={seed}'
            if n > 45:
                # no longer testing cryptominisat
                continue
        else:
            call = '{ex} -rnd-init -rnd-seed={seed}'
        for encoding in os.listdir(solver):
            if not os.path.isdir(os.path.join(solver, encoding)):
                continue
            ifile = '{:03}_{}_{}.dimacs'.format(n, p, q)
            instance = os.path.join(idir, encoding, ifile)
            if not os.path.exists(instance):
                print('skipping non-existing {}'.format(instance), file=sys.stderr)
                continue
            for seed in os.listdir(os.path.join(solver, encoding)):
                if not os.path.isdir(os.path.join(solver, encoding, seed)):
                    continue
                ofile = '{:03}_{}_{}_solution.dimacs'.format(n, p, q)
                solution = os.path.join(solver, encoding, seed, ofile)
                if os.path.exists(solution):
                    print('skipping existing {}'.format(solution), file=sys.stderr)
                    continue
                if n >= 55 and seed != '1000':
                    print('skipping seed {} for {}'.format(seed, solution), file=sys.stderr)
                    continue
                sp = call.format(ex=ex, seed=seed).split()
                print('{}: {} {} ({})'.format(datetime.datetime.now(), solver, instance, seed), file=sys.stderr)
                with open(instance) as fin, open(solution, 'w', buffering=1) as fout:
                    try:
                        w0 = time.perf_counter()
                        u0 = resource.getrusage(resource.RUSAGE_CHILDREN)[0]
                        subprocess.call(sp, stdin=fin, stdout=fout)
                        w1 = time.perf_counter() - w0
                        u1 = resource.getrusage(resource.RUSAGE_CHILDREN)[1] - u0
                    except subprocess.TimeoutExpired as e:
                        w1, u1 = -1, -1
                with open(os.path.join(solver, encoding, seed, 'timing.txt'), 'a', buffering=1) as timing:
                    timing.write('{} {} {} {} {} {}\n'.format(n, p, q, pq, w1, u1))
