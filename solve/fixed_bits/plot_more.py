#!/usr/bin/env python3

from matplotlib import pyplot
import numpy as np

def collect_data(timing_file):
    data = {};
    with open(timing_file) as f:
        for line in f:
            [fname, r, bits, t] = line.split()
            [size, p, q] = fname.split('.')[0].split('_')
            size = int(size)
            if size < 220:
                continue
            p = int(p)
            q = int(q)
            r = int(r)
            bits = int(bits)
            try:
                t = float(t)
            except ValueError:
                # TODO: how to handle this better?
                if size < 220:
                    t = 10
                else:
                    t = 20 

            if size not in data:
                data[size] = {}
            if (p,q) not in data[size]:
                data[size][(p,q)] = []
            if r <= len(data[size][(p,q)]):
                data[size][(p,q)].append({})
            if bits not in data[size][(p,q)][r]:
                data[size][(p,q)][r][bits] = []
            data[size][(p,q)][r][bits].append(t)
    return data

# FIXME: super-slow
def plot_size(size, data):
    for ((p,q), v1) in data.items():
        merged = {}
        for (r, v2) in enumerate(v1):
            xs = []
            ys = []
            for (bits, v3) in v2.items():
                xs.append(bits)
                ys.append(np.median(v3))
            pyplot.plot(xs, ys)
    pyplot.title(f'Size {size}')
    pyplot.xlim(left=0, right=size)
    pyplot.yscale('log')
    pyplot.xlabel('bits fixed')
    pyplot.ylabel('runtime (seconds)')
    pyplot.savefig(f'img/{size}.pdf')
    pyplot.close()

def plot_size_mean(size, data):
    for ((p,q), v1) in data.items():
        merged = {}
        for (r, v2) in enumerate(v1):
            for (bits, v3) in v2.items():
                if bits not in merged:
                    merged[bits] = []
                for t in v3:
                    merged[bits].append(t)
    xs = []
    ys = []
    for bits in sorted(merged):
        xs.append(bits)
        ys.append(np.mean(merged[bits]))
    pyplot.plot(xs, ys)
    pyplot.title(f'Size {size}')
    pyplot.xlim(left=0, right=size)
    pyplot.yscale('log')
    pyplot.xlabel('bits fixed')
    pyplot.ylabel('runtime (seconds)')
    pyplot.savefig(f'img/mean_{size}.pdf')
    pyplot.close()

def plot_sizes(data):
    for (size, d) in data.items():
        plot_size_mean(size, d)
        plot_size(size, d)

def main():
    data = collect_data('timing_more.txt')
    plot_sizes(data)

if __name__ == '__main__':
    main()

