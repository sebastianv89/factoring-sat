#!/usr/bin/env python3

# Transform a .dimacs file into the corresponding Macaulay linear system
# describing the corresponding Boolean equation system, according to
# https://arxiv.org/abs/1712.06239
# For each instance we compute the condition number of the linear system.


# TODO: __imul__, __iadd__: the *= and += operator variations


import sys, os, itertools, datetime
from collections import OrderedDict, Counter
from numbers import Number, Integral
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds
from scipy.io import mmwrite
import numpy as np


def choose(n, k):
    '''Compute the binomial coefficient (n choose k)
    
    Computes exactly with only integer arithmetic and keeping
    internal values relatively small.
    
    Parameters
    ----------
    n : positive int
    k : int
    
    Returns
    -------
    positive int
    
    '''

    k = min(k, n-k)
    if k < 0:
        return 0
    res = 1
    for i in range(1, k+1):
        res *= n-k+i
        res //= i
    return res


# TODO: monomials should not store the dimension, the polynomial should
class Monomial:
    '''Sparse representation of a monomial

    Sparse monomials are represented internally by an OrderedDict(),
    where the key represents the variable and the value represents the
    exponent.  Variables that do not occur in the dictionary represent
    variables with exponent zero.  The dictionary should be ordered by
    key, to ensure the correct and efficient computation of the
    monomial rank, as well as the efficient creation of the sparse
    matrix.

    Attributes
    ----------
    varexps : OrderedDict
        {variable: exponent}, which must always be ordered by variable
    dimensions: int
        dimensions of the multivariate space in which the monomial lives
        (total number of variables)
    degree : int
        degree of the monomial (sum of exponents)

    Notes
    -----
    An OrderedDict only guarantees that the output order is equal to
    the insertion order, the programmer holds responsibility for
    inserting the elements in correct order such that the insertion
    order corresponds with the key-order.

    '''

    def __init__(self, varexps, dimensions, degree=None):
        '''Create a Monomial
        
        Variables not occurring in the terms are assumed to be zero.
        
        Parameters
        ----------
        varexps : OrderedDict
            {variable: exponent}, must be in ascending order of variables
        dimensions : int
            dimensions of the multivariate space in which the monomial lives
        degree : int, optional
            degree of the monomial, will be derived if unspecified (else it
            is the responsibility of the caller to ensure it is correct)

        '''
        self.varexps = varexps
        self.dimensions = dimensions
        if degree is None:
            self.degree = sum(varexps.values())
        else:
            self.degree = degree
    

    def from_monomial(other):
        '''Constructor that copies another monomial'''
        return Monomial(other.varexps.copy(), other.dimensions, other.degree)
    

    def zero(dimensions):
        '''Constructor for zero-degree monomial'''
        return Monomial(OrderedDict(), dimensions)
    
        
    def from_variable(var, dimensions):
        '''Constructor for degree-one monomial'''
        return Monomial(OrderedDict({var: 1}), dimensions)


    def from_dense(exponents):
        '''Create a Monomial from its dense representation.
        
        Parameters
        ----------
        exponents : list of int
            list of exponents, where each index represents the
            corresponding variable

        Returns
        -------
        Monomial
            Monomial in its sparse representation

        '''
        return Monomial(OrderedDict((var, exp) \
                                    for (var, exp) in enumerate(exponents) \
                                    if exp != 0),
                        len(exponents))
    

    def __repr__(self):
        return 'Monomial(' + repr(self.varexps) + ', ' + repr(self.dimensions) + ')'


    def __str__(self):
        if self.is_constant():
            return '1'
        def stringify(var, exp):
            if exp == 1:
                return 'x' + str(var)
            return 'x' + str(var) + '^' + str(exp)
        return ' '.join(stringify(var, exp) for (var, exp) in self.varexps.items())

    
    def _merge_ordereddict(od1, od2):
        # TODO: this is one ugly function, can this really not be improved?
        res = OrderedDict()
        it1, it2 = iter(od1.items()), iter(od2.items())
        k1, k2 = None, None
        try:
            while True:
                # extract next item
                if k1 is None:
                    (k1, v1) = next(it1)
                    if k2 is None:
                        (k2, v2) = next(it2)
                else:
                    assert k2 is None
                    (k2, v2) = next(it2)
                # merge extracted items
                if k1 < k2:
                    res[k1] = v1
                    k1 = None
                elif k1 == k2:
                    res[k1] = v1 + v2
                    k1, k2 = None, None
                else:
                    res[k2] = v2
                    k2 = None
        except StopIteration:
            # add already extracted items
            if k1 is not None:
                res[k1] = v1
            if k2 is not None:
                res[k2] = v2
        # add not-yet-extracted items
        for (k1, v1) in it1:
            res[k1] = v1
        for (k2, v2) in it2:
            res[k2] = v2
        return res
    

    def __eq__(self, other):
        return isinstance(other, Monomial) and \
            self.dimensions == other.dimensions and \
            self.degree == other.degree and \
            self.varexps == other.varexps
    

    def __lt__(self, other):
        if not isinstance(other, Monomial) or self.dimensions != other.dimensions:
            return NotImplemented
        elif self.degree < other.degree:
            return True
        elif self.degree > other.degree:
            return False
        else:
            for ((v1, e1), (v2, e2)) in zip(self.varexps.items(), other.varexps.items()):
                if v1 < v2 or (v1 == v2 and e1 > e2):
                    return True
            return False


    def __mul__(self, rhs):
        if isinstance(rhs, Monomial):
            assert self.dimensions == rhs.dimensions
            if self.is_constant():
                return Monomial(rhs.varexps.copy(), rhs.dimensions)
            return Monomial(Monomial._merge_ordereddict(self.varexps, rhs.varexps), self.dimensions)
        elif isinstance(rhs, Number):
            return Term(rhs, Monomial.from_monomial(self))
        return NotImplemented

    
    def __rmul__(self, lhs):
        return self.__mul__(lhs)
    
        
    def __pow__(self, rhs):
        if isinstance(rhs, Integral):
            return Monomial(OrderedDict((v, rhs*e) for (v, e) in self.varexps.items()),
                            self.dimensions,
                            self.degree * rhs)
        else:
            return NotImplemented


    def is_constant(self):
        '''True if the monomial has no variables (degree == zero)'''
        return self.degree == 0

    
    def drl_rank(self):
        '''Compute the DRL-rank of the monomial.
        
        Compute the rank in a degree reverse lexicographic (DRL)
        ordering.  This order is also known as the graded reverse
        lexicographic order.  Assumes the non-conventional ordering x1
        < x2 < ... < xn of the variables.

        '''
        d = self.degree - 1 # keep track of the "remaining degree"
        dim = self.dimensions
        v_prev = -1
        rank = 0
        for (v, e) in self.varexps.items():
            term = (dim - v_prev) * choose(d + dim - v_prev, dim - v_prev)
            term -= (dim - v) * choose(d + dim - v, dim - v)
            # assert term % (d+1) == 0, 'Non-zero remainder (math is broken?)'
            term //= d + 1
            rank += term
            d -= e
            v_prev = v
        return rank


    def drl_rank_dense(monomial):
        '''Compute the DRL-rank of a dense monomial.
        
        You probably want the regular drl_rank computation, which is more
        efficient for sparse monomials.  (If your monomial is not sparse,
        why are you using this class?)  I don't want to remove this method
        as it might be helpful for testing purposes etc.
        
        Parameters
        ----------
        monomial : list of int
            dense representation of a monomial

        '''
        dim = len(monomial)
        degree = sum(monomial)
        sum_x = 0
        rank = 0
        for i in range(dim):
            rank += choose(dim + degree - 1 - i - sum_x, dim - i)
            sum_x += mono[i]
        return rank
        

    def to_dense(self):
        '''Get the dense representation of the monomial.

        The dense representation is a list of exponents, where the
        variable is left implicit as the index of the corresponding
        exponent.
        
        Returns
        -------
        list of int
            Monomial in its dense representation (a list of exponents)

        '''
        return [self.varexps.get(var, 0) for var in range(self.dimensions)]


def generate_drl_monomials(dimensions, max_degree=None):
    '''Generate all monomials in DRL-order.
    
    This method operates under the non-standard convention that x_1 < x_2 < ... < x_n
    
    Parameters
    ----------
    dimensions : int
        dimensions in which the monomials live (total number of variables)
    max_degree : int, optional
        highest degree of monomials which are generated;  infinite generator if
        no max_degree is specified
    
    Yields
    ------
    Monomial
        the next monomial (generated in DRL-order)

    '''

    # handle trivial cases
    if dimensions<= 0:
        # yield nothing: empty generator
        return
        yield

    # set the boundary condition (if any max_degree was specified)
    if max_degree is None:
        boundary = lambda _: True
    else:
        boundary = lambda mono: mono.get(dimensions-1, 0) < max_degree
    
    # initialize the monomial at zero-degree monomial
    degree = 0
    mono = OrderedDict()
    yield Monomial(mono, dimensions, degree)
    mono[dimensions-1] = 0 # ensures popitem can be called initially

    while boundary(mono):
        # get last non-zero entry
        (j, mj) = mono.popitem()
        
        # update the monomial
        if j < dimensions-1:
            if mj > 1:
                mono[j] = mj - 1
            mono[j+1] = 1
        elif mono: # check if mono is non-empty
            (i, mi) = mono.popitem()
            if mi > 1:
                mono[i] = mi - 1
            mono[i+1] = mj + 1
        else:
            degree += 1
            mono[0] = mj + 1
        
        # yield the current monomial
        yield Monomial(mono, dimensions, degree)
    

class Term:
    '''A (multi-variate) `Polynomial` is a summation of Terms
    
    Attributes
    ----------
    coefficient : number
    monomial : Monomial
    
    '''
    
    def __init__(self, coefficient, monomial):
        '''Create a new Term'''
        self.coefficient = coefficient
        self.monomial = monomial
    

    def constant(coefficient, dimensions):
        return Term(coefficient, Monomial.zero(dimensions))
    
    
    def from_monomial(monomial, coefficient=1):
        '''Create a new Term by copying a Monomial'''
        return Term(coefficient, Monomial.from_monomial(monomial))

    
    def from_term(term):
        '''Create a new Term by copying'''
        return Term(term.coefficient, Monomial.from_monomial(term.monomial))
    

    def degree(self):
        return self.monomial.degree
    

    def __repr__(self):
        return 'Term(' + repr(self.coefficient) + ', ' + repr(self.monomial) + ')'
    
        
    def __str__(self):
        if self.monomial.is_constant():
            return str(self.coefficient)
        return str(self.coefficient) + ' ' + str(self.monomial)
        

    def __mul__(self, rhs):
        if isinstance(rhs, Term):
            return Term(self.coefficient * rhs.coefficient,
                        self.monomial * rhs.monomial)
        elif isinstance(rhs, Monomial):
            return Term(self.coefficient, self.monomial * rhs)
        elif isinstance(rhs, Number):
            return Term(self.coefficient * rhs,
                        Monomial.from_monomial(self.monomial))
        else:
            return NotImplemented
    
    def __rmul__(self, lhs):
        return self.__mul__(self, lhs)

    def __add__(self, rhs):
        '''Adding Terms results in a Polynomial'''
        if isinstance(rhs, Term):
            if self.monomial < rhs.monomial:
                return Polynomial([Term.from_term(self), Term.from_term(rhs)])
            else:
                assert self.monomial == rhs.monomial
                return Polynomial([Term(self.coefficient + rhs.coefficient,
                                        Monomial.from_monomial(self.monomial))])
        else:
            return NotImplemented


class Polynomial:
    '''A (multi-variate) Polynomial.
    
    A Polynomial is a sum of `Term`s.  No duplicate Terms are allowed.  Terms
    should be ordered ascendingly, defined by the ordering of the monomials.
    
    Attributes
    ----------
    terms : list of Term
    '''

    
    def __init__(self, terms):
        self.terms = terms
    

    def from_monomial(monomial, coefficient=1):
        return Polynomial([Term.from_monomial(monomial, coefficient)])

    
    def from_term(term):
        return Polynomial([Term.from_term(term)])


    def from_polynomial(polynomial):
        return Polynomial(list(map(Term.from_term, polynomial.terms)))
    

    def degree(self):
        return max(map(Term.degree, self.terms))

    
    def __repr__(self):
        return 'Polynomial([' + ', '.join(map(repr, self.terms)) + '])'
    
        
    def __str__(self):
        ts = map(str, self.terms)
        return ' + '.join(ts)
    

    def __add__(self, rhs):
        '''Add is very demanding: it requires that monomial/term(s) appended to the list are larger'''
        if isinstance(rhs, Monomial):
            assert all(map(lambda t: t.monomial < rhs, self.terms))
            return Polynomial(self.terms + [Term.from_monomial(rhs)])
        elif isinstance(rhs, Term):
            assert all(map(lambda t: t.monomial < rhs.monomial, self.terms))
            return Polynomial(self.terms + [Term.from_term(rhs)])
        elif isinstance(rhs, Polynomial):
            if rhs.terms != []:
                assert all(map(lambda t: t.monomial < rhs.terms[0].monomial(), self.terms))
            return Polynomial(self.terms + list(map(lambda t: Term.from_term(t), rhs.terms)))
        else:
            return NotImplemented


    def __mul__(self, rhs):
        if isinstance(rhs, Number) or isinstance(rhs, Monomial) or isinstance(rhs, Term):
            return Polynomial(list(map(lambda t: t * rhs, self.terms)))
        elif isinstance(rhs, Polynomial):
            # TODO: optimize (although we won't use it)
            terms = sorted(l * r for l in self.terms for r in rhs.terms)
            terms.sort()
            return Polynomial(sorted(x * y for x in self.terms for y in rhs.terms))
        else:
            return NotImplemented
    
    def to_sparse(self):
        return [(t.monomial.drl_rank(), t.coefficient) for t in self.terms]


# returns number of variables and a list of all clauses
def read_clauses(dimacs_file):
    with open(dimacs_file) as f:
        first_line = f.readline()
        assert first_line.startswith('p cnf'), \
            'Unexpected first line in {}\nExpected "p cnf", got {}'.format(fname, first_line)
        var_count = int(first_line.split()[2])
        clauses = []
        for line in f:
            if line.startswith('c'):
                continue
            clauses.append(list(map(int, line.split()[:-1])))
    return clauses, var_count


# Convert the clauses to the boolean system as defined in the paper
# (Proposition 5.14).  Negated sat-variables are considered a variable
# of their own in this context, so list them in order after the
# non-negated variables.  Start counting at zero:
#   1 -> 0,
#   2 -> 1,
#   ...,
#   n -> n-1,
#   -1 -> n,
#   -2 -> n+1,
#   ...,
#   -n -> 2n-1
def clauses_to_boolean_system(clauses, var_count):
    # TODO: remove these assumptions by making this function more general
    assert max(map(len, clauses)) == 3, 'clauses are not 3SAT'
    assert min(map(len, clauses)) == 3, 'clauses are not exactly-3SAT'
    
    dimensions = 2*var_count
    
    # create equations specifying the clauses
    def clause_to_poly(clause):
        '''A clause is translated to the product of negated variables (a single monomial)'''
        c = Counter()
        for var in clause:
            if var < 0:
                c[-var-1] += 1
            else:
                c[var-1+var_count] += 1
        return Polynomial([Term(1, Monomial(OrderedDict(sorted(c.items())), dimensions))])
    clause_eqs = (clause_to_poly(c) for c in clauses)

    # create equations specifying the variables are binary: x_k^2 - x_k
    def bineq_to_poly(var):
        m = Monomial.from_variable(var, dimensions)
        return -1*m + 1*m**2
    bin_eqs = (bineq_to_poly(x) for x in range(var_count))
    
    # create equations relating SAT-variables with their negation: x_k + not(x_k) - 1
    def negeq_to_poly(var):
        c = Term.constant(-1, dimensions)
        m = Monomial.from_variable(var, dimensions)
        neg_m = Monomial.from_variable(var + var_count, dimensions)
        return c + 1*m + 1*neg_m
    neg_eqs = (negeq_to_poly(x) for x in range(var_count))

    return itertools.chain(clause_eqs, bin_eqs, neg_eqs)


def boolean_system_to_modified_macaulay(eqs, dimensions):
    rows = []
    data = []
    current_degree = 0
    # TODO this line assumes 3SAT, it shouldn't be too hard to generalize this
    for m in generate_drl_monomials(dimensions, 2):
        if m.degree > current_degree:
            current_degree = m.degree
            eqs = list(filter(lambda p: p.degree() + current_degree <= 3, eqs))
        for eq in eqs:
            p_sparse = (eq * m).to_sparse()
            rows.append(list(map(lambda term: term[0], p_sparse)))
            data.append(list(map(lambda term: term[1], p_sparse)))
    # convert to matrix
    nrows = len(rows)
    ncols = choose(dimensions + 3, dimensions)
    # dtype necessary?
    m = lil_matrix((nrows, ncols))
    m.rows = np.array(rows)
    m.data = np.array(data)
    return m


def condition(matrix):
    '''Compute the conditional number of the (sparse) matrix'''
    singular_max = svds(matrix, k=1, which='LM', return_singular_vectors=False)
    singular_min = svds(matrix, k=1, which='SM', return_singular_vectors=False)
    return singular_max[0] / singular_min[0], singular_max, singular_min


solved = []
with open('condition.txt') as fin:
    for line in fin:
        if line.startswith('#'):
            continue
        n, p, q, pq = map(int, line.split()[:4])
        solved.append((p, q))

for line in sys.stdin:
    if line.startswith('#'):
        continue
    n, p, q, pq = map(int, line.split())
    instance_file = '../../instances/long_3sat/{:03}_{}_{}.dimacs'.format(n, p, q)
    if not os.path.exists(instance_file):
        print('skipping non-existing {}'.format(instance), file=sys.stderr)
        continue
    if (p, q) in solved:
        print('skipping solved {} {} {} {}'.format(n, p, q, pq), file=sys.stderr)
        continue
    clauses, var_count = read_clauses(instance_file)
    eqs = list(clauses_to_boolean_system(clauses, var_count))
    m = boolean_system_to_modified_macaulay(eqs, 2*var_count)
    print('{}: {} ({})'.format(datetime.datetime.now(), instance_file, m.shape), file=sys.stderr)
    #mmwrite('./matrices/{:03}_{}_{}_{}.mtx'.format(n, p, q, pq), m)
    c, s1, sn = condition(m)
    print('{} {} {} {} {} {} {}'.format(n, p, q, pq, c, s1, sn, m.shape[0], m.shape[1]))
    sys.stdout.flush()
