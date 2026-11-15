"""Farmer's problem (Birge & Louveaux, Introduction to Stochastic Programming, Sec. 1.1).

Split 500 acres among wheat, corn and sugar beets (planting cost 150, 230, 260 per acre).
After the yield is known: buy or sell wheat/corn to meet 200 t wheat and 240 t corn
of cattle feed; sell beets at 36 per t up to 6000 t, and at 10 per t above that.
Yields (t/acre) are 20% above / at / 20% below average (2.5, 3, 20), each with prob 1/3.
Known optimum: x* = (170, 80, 250), expected profit 108,390.

y = (buy wheat, buy corn, sell wheat, sell corn, sell beets at 36, sell beets at 10)
Second-stage rows, written as W y <= h - T x:
    -buy_w + sell_w       <= yield_w * x_w - 200
    -buy_c + sell_c       <= yield_c * x_c - 240
    sell_b36 + sell_b10   <= yield_b * x_b
"""
import numpy as np
from twosp import TwoStageModel, evpi, vss

inf = np.inf
sp = TwoStageModel(c=np.array([-150, -230, -260]), A=np.array([[1, 1, 1]]), b=np.array([500]),
                   sense="<=", objective="max")
sp.set_second_stage(W=np.array([[-1, 0, 1, 0, 0, 0],
                                [0, -1, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 1]]),
                    ub=np.array([inf, inf, inf, inf, 6000, inf]), sense="<=")

q = np.array([-238, -210, 170, 150, 36, 10])
h = np.array([-200, -240, 0])
for factor in (1.2, 1.0, 0.8):
    yields = factor * np.array([2.5, 3, 20])
    sp.add_scenario(prob=1, q=q, h=h, T=-np.diag(yields))

model = sp.build()
model.optimize()
print("x* =", sp.x.X, " expected profit =", model.ObjVal)
print("EVPI =", evpi(sp), " VSS =", vss(sp))  # book: 7,016 and 1,150
