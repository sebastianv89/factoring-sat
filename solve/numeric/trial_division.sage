# factor semi-primes using trial division

import sys
from sage.rings.factorint import factor_trial_division

ns = []
with open('semiprimes_trial_division.txt') as f:
   for line in f:
      if line.startswith('#'):
         continue
      words = line.split()
      n = sage_eval(words[3])
      ns.append(n)

for line in sys.stdin:
   if line.startswith('#'):
      continue
   words = line.split()
   n = sage_eval(words[3])
   if n in ns:
      sys.stderr.write('skipping ' + str(n) + '\n')
      continue
   sys.stderr.write('solving ' + str(n) + '\n')
   t0 = walltime()
   factor_trial_division(n)
   t1 = walltime(t0)
   words.append(str(t1))
   print(' '.join(words))
   sys.stdout.flush()
