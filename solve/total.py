#!/usr/bin/env python3

total = 0
with open('sat/cryptominisat5/karatsuba/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        total += float(line.split()[4])
with open('sat/cryptominisat5/long/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        total += float(line.split()[4])
with open('sat/maplecomsps/karatsuba/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        for word in line.split()[4:]:
            total += float(word);
with open('sat/maplecomsps/long/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        for word in line.split()[4:]:
            total += float(word);

with open('sat_min/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        total += 100.0 * float(line.split()[5]); # min time spent solving with 100 seeds

with open('sat_multi/timing.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        total += float(line.split()[2]);

print(total)
