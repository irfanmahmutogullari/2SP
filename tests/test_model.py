import numpy as np
import pytest
import scipy.sparse

from twosp import TwoStageModel

q, T = np.array([3.0, 0.5]), np.array([[1.0]])


def newsvendor(A=None, b=None):
    sp = TwoStageModel(c=np.array([1.0]), A=A, b=b, p1=1)
    sp.set_second_stage(W=np.array([[1.0, -1.0]]))
    sp.add_scenario(prob=3, q=q, h=np.array([2.0]), T=T)  # weights 3:3 -> 0.5/0.5
    sp.add_scenario(prob=3, q=q, h=np.array([4.0]), T=T)
    return sp


def test_newsvendor_optimum():
    # By hand: x=2 -> 5.0, x=3 -> 4.75, x=4 -> 4.5, x=5 -> 5.0, so x=4 is optimal.
    sp = newsvendor()
    m = sp.build()
    m.optimize()
    assert sp.x.X[0] == pytest.approx(4.0)
    assert m.ObjVal == pytest.approx(4.5)


def test_first_stage_constraint():
    sp = newsvendor(A=np.array([[1.0]]), b=np.array([3.0]))  # forces x = 3
    m = sp.build()
    m.optimize()
    assert m.ObjVal == pytest.approx(4.75)


def test_deterministic_first_stage_only():
    sp = TwoStageModel(c=np.array([1.0, 2.0]), A=np.array([[1.0, 1.0]]), b=np.array([5.0]))
    m = sp.build()
    m.optimize()
    assert m.ObjVal == pytest.approx(5.0)


def test_nonpositive_probability():
    with pytest.raises(ValueError, match="positive"):
        newsvendor().add_scenario(prob=0, q=q, h=np.array([2.0]), T=T)


@pytest.mark.parametrize("kwargs, match", [
    (dict(q=np.array([3.0]), h=np.array([2.0]), T=T), "q must"),
    (dict(q=q, h=np.array([2.0, 1.0]), T=T), "h must"),
    (dict(q=q, h=2.0, T=T), "h must"),
    (dict(q=q, h=np.array([2.0]), T=np.ones((2, 1))), "T must"),
    (dict(q=q, h=np.array([2.0]), T=np.ones((1, 2))), "T must"),
])
def test_bad_scenario_shapes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        newsvendor().add_scenario(prob=1, **kwargs)


@pytest.mark.parametrize("kwargs, match", [
    (dict(c=np.ones((2, 1))), "c must"),
    (dict(c=np.ones(2), p1=3), "p1 must"),
    (dict(c=np.ones(2), ub=np.ones(3)), "ub must"),
    (dict(c=np.ones(2), A=np.ones((2, 2))), "together"),
    (dict(c=np.ones(2), A=np.ones((2, 3)), b=np.ones(2)), "A must"),
    (dict(c=np.ones(2), A=np.ones((2, 2)), b=np.ones(1)), "b must"),
    (dict(c=np.ones(2), A=np.ones((2, 2)), b=np.ones(2), sense="<"), "sense must"),
    (dict(c=np.ones(2), A=np.ones((2, 2)), b=np.ones(2), sense=["<=", "<=", "<="]), "sense must"),
    (dict(c=np.ones(2), objective="minimize"), "objective must"),
])
def test_bad_first_stage(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TwoStageModel(**kwargs)


@pytest.mark.parametrize("kwargs, match", [
    (dict(W=np.ones(2)), "W must"),
    (dict(W=np.ones((1, 2)), p2=3), "p2 must"),
    (dict(W=np.ones((1, 2)), lb=np.zeros(3)), "lb must"),
    (dict(W=np.ones((1, 2)), sense="=>"), "sense must"),
])
def test_bad_second_stage(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TwoStageModel(c=np.ones(1)).set_second_stage(**kwargs)


def test_scenario_before_second_stage():
    with pytest.raises(ValueError, match="set_second_stage"):
        TwoStageModel(c=np.ones(1)).add_scenario(prob=1, q=q, h=np.array([2.0]), T=T)


def test_W_cannot_change_after_scenarios():
    W = np.array([[1.0, -1.0]])
    sp = TwoStageModel(c=[1.0], p1=1)
    sp.set_second_stage(W=W)
    sp.set_second_stage(W=W)  # still allowed: no scenarios yet
    sp.add_scenario(prob=1, q=q, h=[2.0], T=T)
    with pytest.raises(ValueError, match="cannot be changed"):
        sp.set_second_stage(W=np.array([[1.0, -1.0, 0.0]]))
    with pytest.raises(AttributeError):
        sp.W = np.ones((1, 3))
    with pytest.raises(ValueError, match="read-only"):
        sp.W[0, 0] = 5.0
    W[0, 0] = 5.0  # the model keeps its own copy
    assert sp.W[0, 0] == 1.0


def test_scenario_data_is_copied():
    # Reusing one array for all scenarios must not make them share data.
    sp = TwoStageModel(c=[1.0], p1=1)
    sp.set_second_stage(W=[[1.0, -1.0]])
    h = np.zeros(1)
    for d in (2.0, 4.0):
        h[0] = d
        sp.add_scenario(prob=1, q=q, h=h, T=T)
    assert [s[2][0] for s in sp.scenarios] == [2.0, 4.0]


def test_lists_and_sparse_are_accepted():
    sp = TwoStageModel(c=[1], p1=1)
    sp.set_second_stage(W=scipy.sparse.csr_matrix([[1.0, -1.0]]))
    sp.add_scenario(prob=1, q=[3, 0.5], h=[2], T=[[1]])
    sp.add_scenario(prob=1, q=[3, 0.5], h=[4], T=scipy.sparse.csr_matrix([[1.0]]))
    m = sp.build()
    m.optimize()
    assert m.ObjVal == pytest.approx(4.5)


@pytest.mark.parametrize("make, error, match", [
    (lambda: TwoStageModel(c="abc"), TypeError, "c must contain only numbers"),
    (lambda: TwoStageModel(c=None), TypeError, "c must contain only numbers"),
    (lambda: TwoStageModel(c=[1, np.nan]), ValueError, "c must not contain NaN"),
    (lambda: TwoStageModel(c=[1, 2], A=[[1, 2], [3]], b=[1, 2]), (TypeError, ValueError), "A must"),
    (lambda: TwoStageModel(c=[1, 2], p1=1.0), TypeError, "p1 must be an integer"),
    (lambda: TwoStageModel(c=[1, 2], lb="low"), TypeError, "lb must contain only numbers"),
    (lambda: TwoStageModel(c=[1, 2], ub=np.nan), ValueError, "ub must not contain NaN"),
    (lambda: TwoStageModel(c=[1, 2], A=[[1, 1]], b=[1], sense=1), ValueError, "sense must"),
    (lambda: TwoStageModel(c=[1, 2], objective=["min"]), ValueError, "objective must"),
    (lambda: TwoStageModel(c=[1]).set_second_stage(W=[["a", "b"]]), TypeError, "W must contain only numbers"),
    (lambda: newsvendor().add_scenario(prob="0.5", q=q, h=[2.0], T=T), TypeError, "probability must be a number"),
    (lambda: newsvendor().add_scenario(prob=np.inf, q=q, h=[2.0], T=T), ValueError, "positive and finite"),
    (lambda: newsvendor().add_scenario(prob=1, q=q, h=[np.inf], T=T), ValueError, "h must not contain"),
    (lambda: newsvendor().add_scenario(prob=1, q=q, h=[2.0], T="T"), TypeError, "T must contain only numbers"),
])
def test_bad_input_types(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_max_with_inequalities():
    # Example 1 of SC-96-46 (stochastic multi-knapsack): x* = (0, 4), z* = 61.32.
    sp = TwoStageModel(c=np.array([1.5, 4.0]), p1=2, ub=5, objective="max")
    sp.set_second_stage(W=np.array([[2, 3, 4, 5], [6, 1, 3, 2]]), p2=4, ub=1, sense="<=")
    for xi1 in np.arange(5, 15.5, 0.5):
        for xi2 in np.arange(5, 15.5, 0.5):
            sp.add_scenario(prob=1, q=np.array([16, 19, 23, 28]), h=np.array([xi1, xi2]), T=np.eye(2))
    m = sp.build()
    m.optimize()
    assert sp.x.X == pytest.approx([0, 4])
    assert m.ObjVal == pytest.approx(61.32, abs=0.01)


def test_farmer():
    # Birge & Louveaux farmer's problem: x* = (170, 80, 250), expected profit 108,390.
    inf = np.inf
    sp = TwoStageModel(c=np.array([-150, -230, -260]), A=np.array([[1, 1, 1]]), b=np.array([500]),
                       sense="<=", objective="max")
    sp.set_second_stage(W=np.array([[-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]]),
                        ub=np.array([inf, inf, inf, inf, 6000, inf]), sense="<=")
    for factor in (1.2, 1.0, 0.8):
        sp.add_scenario(prob=1, q=np.array([-238, -210, 170, 150, 36, 10]), h=np.array([-200, -240, 0]),
                        T=-np.diag(factor * np.array([2.5, 3, 20])))
    m = sp.build()
    m.optimize()
    assert sp.x.X == pytest.approx([170, 80, 250])
    assert m.ObjVal == pytest.approx(108390)


def test_sense_per_row():
    # min x1 + x2  s.t.  x1 + x2 >= 2,  x1 - x2 == 1   ->  x = (1.5, 0.5), obj 2
    sp = TwoStageModel(c=np.array([1.0, 1.0]), A=np.array([[1, 1], [1, -1]]), b=np.array([2, 1]),
                       sense=[">=", "=="])
    m = sp.build()
    m.optimize()
    assert sp.x.X == pytest.approx([1.5, 0.5])
