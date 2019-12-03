These instances are just copies of the long multiplication SAT instances, but several of the circuit input bits have been fixed.

The filenames are significant.

## lsb

Fixing the least significant bits of q or both p and q.

`lsb/{size}_{p}_{q}_p{#fixed}_q{#fixed}.dimacs`

## msb

Fixing the most significant bits of q or both p and q.

`msb/{size}_{p}_{q}_p{#fixed}_q{#fixed}.dimacs`

## random

Fixing random bits.

### `random/q_only`

Fixing random bits of q.

`random/q_only/{size}_{p}_{q}_{#fixed}_{index}.dimacs`

`index` has no meaning, but allows to have sample the random bit positions many times.

### `random/both`

Fixing random bits of both p and q.

`random/q_only/{size}_{p}_{q}_{#fixed}_{index}.dimacs`

`index` has no meaning, but allows to have sample the random bit positions many times.

