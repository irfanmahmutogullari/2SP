"""Extensive form of a two-stage stochastic integer program:

    min/max  c'x + sum_k prob_k * q_k'y_k
    s.t.     A x  (sense1)  b
             W y_k + T_k x  (sense2)  h_k  for every scenario k
             lb1 <= x <= ub1, last p1 entries integer
             lb2 <= y_k <= ub2, last p2 entries integer   (default bounds: 0 and +inf)

sense is "==" (or "="), "<=" or ">=": one for all rows, or one per row. Probabilities are scaled to sum to 1.
With no scenarios, this is the deterministic model.

    sp = TwoStageModel(c, A, b)             # first stage
    sp.set_second_stage(W)                  # recourse matrix, the same in every scenario
    sp.add_scenario(prob, q, h, T)          # once per scenario
    model = sp.build()                      # Gurobi model
    model.optimize()                        # solution in sp.x.X and sp.ys[k].X

Vectors and matrices may be lists or NumPy arrays; matrices may also be SciPy sparse.
"""
import numbers
import operator

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy import sparse

SENSES = {"==": operator.eq, "=": operator.eq, "<=": operator.le, ">=": operator.ge}
OBJECTIVES = {"min": GRB.MINIMIZE, "max": GRB.MAXIMIZE}


def _check(ok, msg, error=ValueError):
    if not ok:
        raise error(msg)


def _array(v, name, shape=None, allow_inf=False):
    """v as a float array (sparse stays sparse). In shape, a string stands for any size."""
    if sparse.issparse(v):
        a = v.tocsr()
    else:
        try:
            a = np.asarray(v)
        except ValueError:  # rows of different lengths
            raise ValueError(f"{name} must be a rectangular array of numbers") from None
    _check(a.dtype.kind in "biuf",
           f"{name} must contain only numbers, got a {type(v).__name__} with dtype {a.dtype}", TypeError)
    a = a.astype(float)
    if shape is not None:
        expected = "(" + ", ".join(map(str, shape)) + (",)" if len(shape) == 1 else ")")
        _check(a.ndim == len(shape) and all(isinstance(s, str) or s == d for s, d in zip(shape, a.shape)),
               f"{name} must have shape {expected}, got {a.shape}")
    values = a.data if sparse.issparse(a) else a
    if allow_inf:
        _check(not np.isnan(values).any(), f"{name} must not contain NaN")
    else:
        _check(np.isfinite(values).all(), f"{name} must not contain NaN or infinite values")
    return a


def _bound(v, name, n):
    a = _array(v, name, allow_inf=True)
    _check(a.ndim == 0 or a.shape == (n,), f"{name} must be a number or have shape ({n},), got {a.shape}")
    return a


def _count(p, name, n):
    _check(isinstance(p, numbers.Integral), f"{name} must be an integer, got {p!r}", TypeError)
    _check(0 <= p <= n, f"{name} must be between 0 and {n}, got {p}")
    return int(p)


def _senses(sense, m):
    """One sense per row, as an array of length m."""
    s = np.asarray(sense)
    _check(s.ndim == 0 or s.shape == (m,), f"sense must be one value or have shape ({m},), got {s.shape}")
    bad = [v for v in s.ravel().tolist() if not isinstance(v, str) or v not in SENSES]
    _check(not bad, f"sense must be one of {list(SENSES)}, got {bad}")
    return np.broadcast_to(s, (m,))


def _add_rows(m, senses, lhs, rhs):
    """Add lhs[i] (senses[i]) rhs[i] for every row i."""
    for s in set(senses):
        idx = np.flatnonzero(senses == s)
        m.addConstr(SENSES[s](lhs[idx], rhs[idx]))


class TwoStageModel:
    def __init__(self, c, A=None, b=None, p1=0, lb=0, ub=GRB.INFINITY,
                 sense="==", objective="min"):
        """First stage: costs c, constraints A x (sense) b, bounds lb <= x <= ub, last p1 entries of x integer."""
        self.c = _array(c, "c", ("n1",))
        n1 = len(self.c)
        self.p1 = _count(p1, "p1", n1)
        self.lb1, self.ub1 = _bound(lb, "lb", n1), _bound(ub, "ub", n1)
        _check(isinstance(objective, str) and objective in OBJECTIVES,
               f"objective must be 'min' or 'max', got {objective!r}")
        self.objective = objective
        _check((A is None) == (b is None), "A and b must be given together")
        self.A = self.b = None
        if A is not None:
            self.A = _array(A, "A", ("m1", n1))
            self.b = _array(b, "b", (self.A.shape[0],))
        self.sense1 = _senses(sense, 0 if A is None else self.A.shape[0])
        self._W, self.p2 = None, 0
        self.scenarios = []

    @property
    def W(self):
        """Recourse matrix (read-only): set it with set_second_stage, before any scenario is added."""
        return self._W

    def set_second_stage(self, W, p2=0, lb=0, ub=GRB.INFINITY, sense="=="):
        """Second stage, the same in every scenario: recourse matrix W, constraint sense,
        bounds lb <= y <= ub, last p2 entries of y integer."""
        _check(not self.scenarios, "the second stage (W) cannot be changed after scenarios have been added")
        W = _array(W, "W", ("m2", "n2"))
        if not sparse.issparse(W):
            W.flags.writeable = False
        m2, n2 = W.shape
        p2 = _count(p2, "p2", n2)
        lb, ub = _bound(lb, "lb", n2), _bound(ub, "ub", n2)
        self.sense2 = _senses(sense, m2)
        self._W, self.p2, self.lb2, self.ub2 = W, p2, lb, ub

    def add_scenario(self, prob, q, h, T):
        """Scenario with weight prob > 0, second-stage costs q and constraints W y (sense) h - T x."""
        k = len(self.scenarios)
        _check(self.W is not None, "call set_second_stage before add_scenario")
        _check(isinstance(prob, numbers.Real), f"scenario {k}: probability must be a number, got {prob!r}", TypeError)
        _check(0 < prob < np.inf, f"scenario {k}: probability must be positive and finite, got {prob}")
        (m2, n2), n1 = self.W.shape, len(self.c)
        self.scenarios.append((float(prob),
                               _array(q, f"scenario {k}: q", (n2,)),
                               _array(h, f"scenario {k}: h", (m2,)),
                               _array(T, f"scenario {k}: T", (m2, n1))))

    def build(self):
        """Returns the Gurobi model; variables are in self.x and self.ys."""
        n1 = len(self.c)
        m = gp.Model()

        self.x = m.addMVar(n1, vtype=[GRB.CONTINUOUS] * (n1 - self.p1) + [GRB.INTEGER] * self.p1,
                           lb=self.lb1, ub=self.ub1)
        if self.A is not None:
            _add_rows(m, self.sense1, self.A @ self.x, self.b)

        obj = self.c @ self.x
        self.ys = []
        total = sum(prob for prob, _, _, _ in self.scenarios)
        for prob, q, h, T in self.scenarios:
            n2 = self.W.shape[1]
            y = m.addMVar(n2, vtype=[GRB.CONTINUOUS] * (n2 - self.p2) + [GRB.INTEGER] * self.p2,
                          lb=self.lb2, ub=self.ub2)
            _add_rows(m, self.sense2, self.W @ y, h - T @ self.x)
            obj += (prob / total) * (q @ y)
            self.ys.append(y)

        m.setObjective(obj, OBJECTIVES[self.objective])
        return m
