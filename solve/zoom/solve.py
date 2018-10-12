import sys, os
import datetime, time
from enum import Enum
import itertools
import subprocess

class Duplicate(Enum):
    ask = 1
    skip = 2
    overwrite = 3

DUPLICATE = Duplicate.ask

def get_semiprimes(fname='./semiprimes35.txt'):
    '''Extract [(n, p, q, pq)] from text file'''
    sps = []
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            sps.append(tuple(map(int, line.split())))
    return sps

def skip_duplicate(ofname):
    '''Should we skip this instance (if duplicate)?  Prompt user and update setting.'''
    global DUPLICATE
    if os.path.exists(ofname):
        if DUPLICATE == Duplicate.ask:
            while True:
                ch = input('Solution file ({}) exists, overwrite?\n'
                            '[y]es, [n]o, [o]verwrite all, [s]kip all? '.format(ofname))[0]
                if ch == 'y':
                    return False
                elif ch == 'n':
                    return True
                elif ch == 'o':
                    DUPLICATE = Duplicate.overwrite
                    return False
                elif ch == 's':
                    DUPLICATE = Duplicate.skip
                    return True
                print('Invalid answer')
        elif DUPLICATE == Duplicate.skip:
            return True
        assert DUPLICATE == Duplicate.overwrite, 'Unknown value in DUPLICATE'
        return False
    return False


def solve_semiprimes(sps, seed_max=None,
        tfile = './timing_long_35.txt',
        exec_format = './maplecomsps -rnd-init -rnd-seed={s}',
        idir = '../../instances/long',
        ifile_format = '{n:03}_{p}_{q}.dimacs',
        odir = './solutions_long/',
        ofile_format = '{n:03}_{p}_{q}_{s:04}_solution.dimacs'):
    '''Solve each semiprime instances using different seeds'''
    assert os.path.isdir(idir), '{} is not a directory'.format(idir)
    assert os.path.isdir(odir), '{} is not a directory'.format(odir)
    if seed_max is None:
        seeds = itertools.count(1)
    else:
        seeds = range(1, seed_max)
    for seed in seeds:
        for (n, p, q, pq) in sps:
            ifname = os.path.join(idir, ifile_format.format(n=n, p=p, q=q))
            assert os.path.exists(ifname), 'input file {} does not exist'.format(ifname)
            ofname = os.path.join(odir, ofile_format.format(n=n, p=p, q=q, s=seed))
            if skip_duplicate(ofname):
                print('Skipping {}'.format(ofname))
                continue
            call = exec_format.format(s=seed).split()
            print('{t}: {i} ({s})'.format(t=datetime.datetime.now(), i=ifname, s=seed))
            with open(ifname) as fin, open(ofname, 'w', buffering=1) as fout:
                t0 = time.perf_counter()
                subprocess.call(call, stdin=fin, stdout=fout)
                t1 = time.perf_counter()
            with open(tfile, 'a', buffering=1) as ft:
                ft.write('{n} {p} {q} {pq} {s} {t}\n'.format(n=n, p=p, q=q, pq=pq, s=seed, t=t1-t0))

def main():
    sps = get_semiprimes()
    solve_semiprimes(sps)

if __name__ == '__main__':
    main()

