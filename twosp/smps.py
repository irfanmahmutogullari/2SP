"""Read a two-stage problem in SMPS format (CORE, TIME and STOCH files) into a TwoStageModel.

Supported subset of SMPS:
  CORE   any MPS file Gurobi can read (no RANGES)
  TIME   PERIODS, implicit form: the first column and row of each of the two stages
  STOCH  SCENARIOS DISCRETE, every scenario's parent is ROOT. A scenario may replace
         right-hand sides (h), coefficients of first-stage columns in second-stage rows (T)
         and second-stage costs (q). W must stay deterministic.

Within each stage the columns are reordered so that the integer ones come last, as
TwoStageModel requires; the returned names give the order of sp.x and sp.ys[k].
"""
import os
import shutil
import tempfile

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .model import TwoStageModel, _check

SENSE = {"<": "<=", ">": ">=", "=": "=="}


def _number(text, path):
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{path}: expected a number, got {text!r}") from None


def _lines(path):
    """Non-empty, non-comment lines, split into fields with quotes removed."""
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("*"):
                yield line[0].isspace(), line.replace("'", " ").split()


def _objective_row(cor):
    """Name of the first N row in the core file's ROWS section."""
    section = None
    for indented, fields in _lines(cor):
        if not indented:
            section = fields[0]
        elif section == "ROWS" and fields[0] == "N":
            return fields[1]
    raise ValueError(f"{cor}: no objective (N) row found")


def _read_core(cor):
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "core.mps")  # Gurobi picks the format from the extension
        shutil.copy(cor, path)
        m = gp.read(path)
    finally:
        shutil.rmtree(tmp)
    _check(m.ObjCon == 0, f"{cor}: objective constants are not supported")
    vs, cs = m.getVars(), m.getConstrs()
    _check(len(cs) == m.NumConstrs and m.NumQConstrs == 0, f"{cor}: only linear constraints are supported")
    return dict(
        A=m.getA().toarray(),
        cols=[v.VarName for v in vs],
        rows=[c.ConstrName for c in cs],
        obj=np.array(m.getAttr("Obj", vs)),
        lb=np.array(m.getAttr("LB", vs)),
        ub=np.array(m.getAttr("UB", vs)),
        integer=np.array([t != GRB.CONTINUOUS for t in m.getAttr("VType", vs)]),
        rhs=np.array(m.getAttr("RHS", cs)),
        sense=np.array([SENSE[s] for s in m.getAttr("Sense", cs)]),
        objective="min" if m.ModelSense == GRB.MINIMIZE else "max",
    )


def _read_time(tim, cols, rows):
    """Index of the first column and first row of stage 2."""
    periods = []
    section = None
    for indented, fields in _lines(tim):
        if not indented:
            section = fields[0]
        elif section == "PERIODS":
            _check(len(fields) == 3, f"{tim}: only the implicit PERIODS form is supported, got {fields}")
            periods.append(fields)
    _check(len(periods) == 2, f"{tim}: expected exactly 2 periods, got {len(periods)}")
    starts = []
    for a, b, _ in periods:  # the standard order is (column, row), but some files swap them
        col, row = (a, b) if a in cols else (b, a)
        _check(col in cols and row in rows, f"{tim}: {a!r}, {b!r} are not a column and a row of the core")
        starts.append((cols.index(col), rows.index(row)))
    _check(starts[0] == (0, 0), f"{tim}: stage 1 must start at the first column and row")
    return starts[1][0], starts[1][1], periods[1][2]


def read_smps(cor, tim, sto):
    """Returns (sp, x_names, y_names), where sp is a TwoStageModel."""
    core = _read_core(cor)
    obj_row = _objective_row(cor)
    j2, i2, stage2 = _read_time(tim, core["cols"], core["rows"])
    A, cols, rows = core["A"], core["cols"], core["rows"]
    _check(not A[:i2, j2:].any(), f"{cor}: stage-1 rows must not contain stage-2 columns")

    # continuous columns first, integer columns last (stable within each stage)
    perm1 = np.argsort(core["integer"][:j2], kind="stable")
    perm2 = j2 + np.argsort(core["integer"][j2:], kind="stable")
    pos1 = {cols[j]: k for k, j in enumerate(perm1)}
    pos2 = {cols[j]: k for k, j in enumerate(perm2)}
    row2 = {rows[i]: i - i2 for i in range(i2, len(rows))}

    sp = TwoStageModel(
        c=core["obj"][perm1],
        A=A[:i2][:, perm1],
        b=core["rhs"][:i2],
        p1=int(core["integer"][:j2].sum()),
        lb=core["lb"][perm1], ub=core["ub"][perm1],
        sense=core["sense"][:i2],
        objective=core["objective"],
    )
    sp.set_second_stage(
        W=A[i2:][:, perm2],
        p2=int(core["integer"][j2:].sum()),
        lb=core["lb"][perm2], ub=core["ub"][perm2],
        sense=core["sense"][i2:],
    )
    q0, h0, T0 = core["obj"][perm2], core["rhs"][i2:], A[i2:][:, perm1]

    def change(scen, col, row, value):
        prob, q, h, T = scen
        if row == obj_row:
            _check(col in pos2, f"{sto}: random cost of {col!r} is only supported for stage-2 columns")
            q[pos2[col]] = value
        else:
            _check(row in row2, f"{sto}: random entries are only supported in stage-2 rows, got row {row!r}")
            _check(col not in pos2, f"{sto}: entry ({col!r}, {row!r}) would make W random, which is not supported")
            if col in pos1:
                T[row2[row], pos1[col]] = value
            else:  # col is the name of the RHS vector
                h[row2[row]] = value

    scenarios = []
    section = None
    for indented, fields in _lines(sto):
        if not indented:
            section = fields[0]
            _check(section in ("STOCH", "SCENARIOS", "ENDATA"),
                   f"{sto}: section {section} is not supported (only SCENARIOS)")
            continue
        _check(section == "SCENARIOS", f"{sto}: data outside the SCENARIOS section")
        if fields[0] == "SC":
            _check(len(fields) == 5, f"{sto}: an SC line needs name, parent, probability and stage, got {fields}")
            _, name, parent, prob, stage = fields
            _check(parent == "ROOT", f"{sto}: scenario {name} has parent {parent}, expected ROOT (two stages)")
            _check(stage == stage2, f"{sto}: scenario {name} starts in {stage}, expected {stage2}")
            scenarios.append((_number(prob, sto), q0.copy(), h0.copy(), T0.copy()))
        else:
            _check(scenarios, f"{sto}: data before the first SC line")
            col, pairs = fields[0], fields[1:]
            _check(len(pairs) in (2, 4), f"{sto}: cannot read line {fields}")
            for row, value in zip(pairs[::2], pairs[1::2]):
                change(scenarios[-1], col, row, _number(value, sto))

    _check(scenarios, f"{sto}: no scenarios found")
    for scen in scenarios:
        sp.add_scenario(*scen)
    return sp, [cols[j] for j in perm1], [cols[j] for j in perm2]
