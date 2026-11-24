import numpy as np
import pytest

from twosp import TwoStageModel, extensive_matrix, plot_structure


def farmer_like():
    sp = TwoStageModel(c=[1.0, 2.0], A=[[1.0, 1.0]], b=[3.0], sense="<=")
    sp.set_second_stage(W=[[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
    for k in range(3):
        sp.add_scenario(prob=1, q=[1.0, 1.0, 1.0], h=[k, 1.0], T=[[k + 1.0, 0.0], [0.0, 2.0]])
    return sp


def test_extensive_matrix_matches_gurobi():
    sp = farmer_like()
    m = sp.build()
    m.update()
    M = extensive_matrix(sp)
    assert M.shape == (1 + 3 * 2, 2 + 3 * 3)
    assert np.allclose(M.toarray(), m.getA().toarray())


def test_extensive_matrix_first_stage_only():
    assert extensive_matrix(TwoStageModel(c=[1.0, 2.0])).shape == (0, 2)


def test_plot_structure():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_structure(farmer_like())
    assert ax.get_title() == "7 rows, 11 columns, 20 nonzeros"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["first stage", "scenario 1", "scenario 2", "scenario 3"]
