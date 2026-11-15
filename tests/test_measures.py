import numpy as np
import pytest

from twosp import (TwoStageModel, recourse_problem, wait_and_see, expected_value_problem,
                   expected_result_of_ev, evpi, vss)


def farmer():
    inf = np.inf
    sp = TwoStageModel(c=np.array([-150, -230, -260]), A=np.array([[1, 1, 1]]), b=np.array([500]),
                       sense="<=", objective="max")
    sp.set_second_stage(W=np.array([[-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]]),
                        ub=np.array([inf, inf, inf, inf, 6000, inf]), sense="<=")
    for factor in (1.2, 1.0, 0.8):
        sp.add_scenario(prob=1, q=np.array([-238, -210, 170, 150, 36, 10]), h=np.array([-200, -240, 0]),
                        T=-np.diag(factor * np.array([2.5, 3, 20])))
    return sp


def test_farmer():
    # Birge & Louveaux, Sec. 1.1 and Ch. 4 (profits, i.e. a max problem):
    # RP = 108,390, WS = 115,406, EVPI = 7,016 (the book rounds WS = 115,405.56); EV = 118,600 at x = (120, 80, 300), EEV = 107,240, VSS = 1,150.
    sp = farmer()
    assert recourse_problem(sp) == pytest.approx(108390)
    assert wait_and_see(sp) == pytest.approx(115406, abs=1)
    ev, x_bar = expected_value_problem(sp)
    assert ev == pytest.approx(118600)
    assert x_bar == pytest.approx([120, 80, 300])
    assert expected_result_of_ev(sp) == pytest.approx(107240)
    assert evpi(sp) == pytest.approx(7016, abs=1)
    assert vss(sp) == pytest.approx(1150)


def test_integer_newsvendor():
    # By hand: scenarios d = 2, 4 (prob 1/2). WS = (2 + 4)/2 = 3, RP = 4.5 at x = 4  ->  EVPI = 1.5.
    # Mean demand 3 -> x_bar = 3, EV = 3; EEV = cost of x = 3 = 4.75  ->  VSS = 0.25.
    sp = TwoStageModel(c=np.array([1.0]), p1=1)
    sp.set_second_stage(W=np.array([[1.0, -1.0]]))
    for d in (2.0, 4.0):
        sp.add_scenario(prob=1, q=np.array([3.0, 0.5]), h=np.array([d]), T=np.array([[1.0]]))
    assert wait_and_see(sp) == pytest.approx(3.0)
    assert expected_value_problem(sp)[0] == pytest.approx(3.0)
    assert expected_result_of_ev(sp) == pytest.approx(4.75)
    assert evpi(sp) == pytest.approx(1.5)
    assert vss(sp) == pytest.approx(0.25)


def test_no_scenarios():
    with pytest.raises(ValueError, match="no scenarios"):
        evpi(TwoStageModel(c=[1.0]))


def test_ev_solution_infeasible_gives_infinite_vss():
    # x must cover demand exactly (no recourse slack in one direction): y = d - x >= 0 forces x <= d.
    # Mean demand 3 gives x_bar = 3, which is infeasible for d = 2, so EEV = +inf.
    sp = TwoStageModel(c=np.array([-1.0]))
    sp.set_second_stage(W=np.array([[1.0]]))
    for d in (2.0, 4.0):
        sp.add_scenario(prob=1, q=np.array([0.0]), h=np.array([d]), T=np.array([[1.0]]))
    assert expected_result_of_ev(sp) == np.inf
    assert vss(sp) == np.inf
