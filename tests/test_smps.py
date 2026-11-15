"""SMPS reader tests on SIPLIB instances (https://www2.isye.gatech.edu/~sahmed/siplib)."""
from pathlib import Path

import pytest

from twosp import read_smps

DATA = Path(__file__).parent / "data"


def solve(sp):
    m = sp.build()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0
    m.optimize()
    return m


def test_sizes10():
    # SIPLIB SIZES10: optimal value 224564.3
    sp, _, _ = read_smps(DATA / "sizes.cor", DATA / "sizes.tim", DATA / "sizes10.sto")
    assert len(sp.scenarios) == 10
    assert solve(sp).ObjVal == pytest.approx(224564.3, abs=0.01)


def test_sizes5_with_siplib_typo_corrected():
    # SIPLIB SIZES5: optimal value 224486. Every scenario scales all demands by one factor,
    # but in the factor-1.0 scenario of sizes5.sto rows D06..D09 are scaled by 1.1. The
    # published value (and SIPLIB's own sizes5.mps) uses 1.0, so we correct that here.
    sp, _, _ = read_smps(DATA / "sizes.cor", DATA / "sizes.tim", DATA / "sizes5.sto")
    sp.scenarios[2][2][5:9] = sp.b[5:9]
    assert solve(sp).ObjVal == pytest.approx(224486.0, abs=0.01)


@pytest.mark.parametrize("name, value", [("sslp_5_25_50", -121.60), ("sslp_15_45_10", -260.50)])
def test_sslp(name, value):
    sp, _, _ = read_smps(DATA / f"{name}.cor", DATA / f"{name}.tim", DATA / f"{name}.sto")
    assert solve(sp).ObjVal == pytest.approx(value, abs=0.01)


def test_names_and_integer_order():
    sp, x_names, y_names = read_smps(DATA / "sizes.cor", DATA / "sizes.tim", DATA / "sizes3.sto")
    assert len(x_names) == len(sp.c) and len(y_names) == sp.W.shape[1]
    assert all(n.endswith("01") for n in x_names) and all(n.endswith("02") for n in y_names)  # names end in the stage
    assert all(n.startswith("Z") for n in x_names[-sp.p1:])  # the integer Z columns come last


def test_random_W_is_rejected(tmp_path):
    sto = tmp_path / "bad.sto"
    sto.write_text("STOCH\nSCENARIOS     DISCRETE\n SC S1  ROOT  1.0  STAGE-2\n    Z01JJ02   D01JJ02   1.0\nENDATA\n")
    with pytest.raises(ValueError, match="W random"):
        read_smps(DATA / "sizes.cor", DATA / "sizes.tim", sto)


def test_indep_is_rejected(tmp_path):
    sto = tmp_path / "bad.sto"
    sto.write_text("STOCH\nINDEP         DISCRETE\n    RHS1   D01JJ02   1.0   0.5\nENDATA\n")
    with pytest.raises(ValueError, match="not supported"):
        read_smps(DATA / "sizes.cor", DATA / "sizes.tim", sto)
