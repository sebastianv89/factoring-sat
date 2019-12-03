#!/usr/bin/env python3

from matplotlib import pyplot
import numpy as np
from copy import deepcopy

def collect_data(timing_file):
    ''' produce two dictionaries:
    {
        <size>: {
            <(p, q)>: [
                <bits>: [
                    <trial>: time
                ]
            ]
        },
    }
    '''
    q_only = {}
    both = {}
    with open(timing_file) as f:
        for line in f:
            [fname, time] = line.split()
            words = fname.split('.')[0].split('_')
            size = int(words[0])
            p = int(words[1])
            q = int(words[2])
            p_bits = int(words[3][1:])
            q_bits = int(words[4][1:])
            t = float(time)

            if p_bits == 0:
                data = q_only
            else:
                data = both
            if size not in data:
                data[size] = {}
            if (p, q) not in data[size]:
                data[size][(p,q)] = [None]
            if len(data[size][(p,q)]) <= q_bits:
                data[size][(p,q)].append([])
            data[size][(p,q)][q_bits].append(t)
    return q_only, both

def collect_random_data(timing_file):
    ''' produce a dictionary:
    {
        <size>: {
            <(p,q)>: [
                <r>: [
                    <bits>: [
                        <trial>: time
                    ]
                ]
            ]
        }
    }
    '''
    data = {}
    with open(timing_file) as f:
        for line in f:
            [fname, time] = line.split()
            words = fname.split('.')[0].split('_')
            size = int(words[0])
            p = int(words[1])
            q = int(words[2])
            r = int(words[3])
            bits = int(words[4])
            t = float(time)

            if size not in data:
                data[size] = {}
            if (p, q) not in data[size]:
                data[size][(p,q)] = []
            if len(data[size][(p,q)]) <= r:
                data[size][(p,q)].append([None])
            if len(data[size][(p,q)][r]) <= bits:
                data[size][(p,q)][r].append([])
            data[size][(p,q)][r][bits].append(t)
    return data

def collect_nobits(timing_file, ref):
    sizes, pqs = set(), set()
    for x in ref.values():
        for y in x.values():
            for (size, z) in y.items():
                sizes.add(size)
                for pq in z:
                    pqs.add(pq)
    data = {}
    with open(timing_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            words = line.split()
            size = int(words[0])
            p = int(words[1])
            q = int(words[2])
            t = float(words[4])

            if size not in sizes:
                continue
            if (p,q) not in pqs:
                continue

            if size not in data:
                data[size] = {}
            if (p,q) not in data[size]:
                data[size][(p,q)] = t
    return data

def collect_all():
    data = {'lsb': {}, 'msb': {}, 'random': {}}
    data['lsb']['q'], data['lsb']['both'] = collect_data('timing_lsb.txt')
    data['msb']['q'], data['msb']['both'] = collect_data('timing_msb.txt')
    data['random']['q'] = collect_random_data('timing_random_q.txt')
    data['random']['both'] = collect_random_data('timing_random_both.txt')
    nobits = collect_nobits('timing_median.txt', data)
    return data, nobits

def compute_medians(data, nobits):
    m = {}
    for (k1, v1) in data.items():
        m[k1] = {}
        for (k2, v2) in v1.items():
            m[k1][k2] = {}
            for (size, v3) in v2.items():
                m[k1][k2][size] = {}
                for (pq, v4) in v3.items():
                    if k1 == 'random':
                        m[k1][k2][size][pq] = []
                        for v5 in v4:
                            assert (v5[0] is None)
                            m[k1][k2][size][pq].append([nobits[size][pq]] + [np.median(v6) for v6 in v5[1:]])
                    else:
                        assert (v4[0] is None)
                        m[k1][k2][size][pq] = [nobits[size][pq]] + [np.median(v5) for v5 in v4[1:]]
    return m

def plot_one(title, xs, ys):
    fig = pyplot.figure()
    ax = fig.subplots()
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel('# bits set')
    ax.set_ylabel('solver time (median)')
    ax.set_yscale('log')
    pyplot.show()
    pyplot.close(fig)

def plot_random_bits(title, data):
    fig = pyplot.figure()
    ax = fig.subplots()
    for r in data:
        ax.plot(r)
    ax.set_title(title)
    ax.set_xlabel('# bits set')
    ax.set_ylabel('solver time (median)')
    ax.set_yscale('log')
    pyplot.show()
    pyplot.close(fig)

def plot_many(title, data, both):
    fig = pyplot.figure()
    ax = fig.subplots()
    for (pq, d) in data.items():
        if both:
            ax.plot(list(range(0, 2*len(d), 2)), d)
        else:
            ax.plot(d)
    ax.set_title(title)
    ax.set_xlabel('# bits set ratio')
    ax.set_ylabel('solver time ratio (median)')
    ax.set_yscale('log')
    pyplot.show()
    pyplot.close(fig)

def plot_all(data, size):
    fig = pyplot.figure(figsize=(8.5, 11), dpi=200)
    pyplot.subplots_adjust(left=.1, right=.95, bottom=.05, top=.95, wspace=.3, hspace=.3)
    index = 1
    for k1 in data:
        for k2 in data[k1]:
            ax = fig.add_subplot(3, 2, index)
            for (s, v3) in data[k1][k2].items():
                if s != size:
                    continue
                for (pq, v4) in v3.items():
                    if k1 == 'random':
                        for r in v4:
                            ax.plot(r)
                    else:
                        ys = [t for t in v4]
                        if k2 == 'q':
                            xs = range(len(ys))
                        else:
                            xs = range(0, 2*len(ys), 2)
                        ax.plot(xs, ys)
            ax.set_title(f'{k1} {k2}')
            ax.set_xlabel('# bits set')
            ax.set_ylabel('solver time (seconds)')
            ax.set_yscale('log')
            index += 1
    pyplot.savefig(f'img/{size}.pdf')
    pyplot.close(fig)

def main():
    data, nobits = collect_all()
    medians = compute_medians(data, nobits)
    for size in range(20, 46):
        plot_all(medians, size)
    return

if __name__ == '__main__':
    main()

