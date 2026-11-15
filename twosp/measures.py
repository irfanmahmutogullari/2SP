"""Value of information and of the stochastic solution (Birge & Louveaux, Chapter 4).

    RP   = min_x E[z(x, xi)]              recourse problem (here-and-now)
    WS   = E[min_x z(x, xi)]              wait-and-see: each scenario solved on its own
    EV   = min_x z(x, mean xi), solution x_bar      expected value problem
    EEV  = E[z(x_bar, xi)]                expected result of using x_bar
    EVPI = RP - WS,   VSS = EEV - RP      (signs flipped for "max", so both are >= 0)
"""
import copy

import numpy as np
from gurobipy import GRB


def _solve(sp):
    """(optimal value, first-stage solution) of sp; value is +inf (min) or -inf (max) if infeasible."""
    m = sp.build()
    m.Params.OutputFlag = 0
    m.optimize()
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return (np.inf if sp.objective == "min" else -np.inf), None
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi stopped with status {m.Status}")
    return m.ObjVal, sp.x.X.copy()


def _check_finite(value, msg):
    if not np.isfinite(value):
        raise ValueError(msg)


def _with_scenarios(sp, scenarios):
    s = copy.copy(sp)
    s.scenarios = scenarios
    return s


def _probabilities(sp):
    """Scenario probabilities, scaled to sum to 1."""
    if not sp.scenarios:
        raise ValueError("the model has no scenarios: call add_scenario first")
    probs = np.array([s[0] for s in sp.scenarios])
    return probs / probs.sum()


def _sign(sp):
    return 1 if sp.objective == "min" else -1


def recourse_problem(sp):
    """RP: optimal value of the stochastic program."""
    return _solve(sp)[0]


def wait_and_see(sp):
    """WS: expected optimal value when the scenario is known before deciding x."""
    ws = 0.0
    for k, (p, s) in enumerate(zip(_probabilities(sp), sp.scenarios)):
        value = _solve(_with_scenarios(sp, [s]))[0]
        _check_finite(value, f"scenario {k} has no feasible solution on its own")
        ws += p * value
    return ws


def expected_value_problem(sp):
    """EV and x_bar: optimal value and first-stage solution for the mean scenario."""
    probs = _probabilities(sp)
    q, h, T = (sum(p * s[i] for p, s in zip(probs, sp.scenarios)) for i in (1, 2, 3))
    value, x_bar = _solve(_with_scenarios(sp, [(1.0, q, h, T)]))
    _check_finite(value, "the expected value problem is infeasible")
    x_bar[len(x_bar) - sp.p1:] = np.round(x_bar[len(x_bar) - sp.p1:])  # exact integers
    return value, x_bar


def expected_result_of_ev(sp):
    """EEV: expected value of fixing the first stage to the EV solution x_bar."""
    _, x_bar = expected_value_problem(sp)
    fixed = copy.copy(sp)
    fixed.lb1 = fixed.ub1 = x_bar
    return _solve(fixed)[0]


def evpi(sp):
    """Expected value of perfect information: RP - WS (min) or WS - RP (max)."""
    return _sign(sp) * (recourse_problem(sp) - wait_and_see(sp))


def vss(sp):
    """Value of the stochastic solution: EEV - RP (min) or RP - EEV (max)."""
    return _sign(sp) * (expected_result_of_ev(sp) - recourse_problem(sp))
