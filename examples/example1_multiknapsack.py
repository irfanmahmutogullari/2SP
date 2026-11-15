"""Stochastic multi-knapsack: Example 1 of Carøe & Schultz, "Dual decomposition in stochastic
integer programming" (ZIB preprint SC 96-46, 1996), adapted there from Schultz, Stougie & van der Vlerk.

    max  3/2 x1 + 4 x2 + E[Q(x, xi)],   0 <= x1, x2 <= 5 integer
    Q(x, xi) = max 16 y1 + 19 y2 + 23 y3 + 28 y4
               s.t. 2 y1 + 3 y2 + 4 y3 + 5 y4 <= xi1 - x1
                    6 y1 +   y2 + 3 y3 + 2 y4 <= xi2 - x2
                    y binary
    xi uniform on {5, 5.5, ..., 15}^2  (441 scenarios).
    Paper reports: x* = (0, 4), z* = 61.32.
"""
import numpy as np
from twosp import TwoStageModel

sp = TwoStageModel(c=np.array([1.5, 4.0]), p1=2, ub=5, objective="max")
sp.set_second_stage(W=np.array([[2, 3, 4, 5],
                                [6, 1, 3, 2]]), p2=4, ub=1, sense="<=")

q = np.array([16, 19, 23, 28])
T = np.eye(2)
for xi1 in np.arange(5, 15.5, 0.5):
    for xi2 in np.arange(5, 15.5, 0.5):
        sp.add_scenario(prob=1, q=q, h=np.array([xi1, xi2]), T=T)

model = sp.build()
model.optimize()
print("x* =", sp.x.X, " z* =", model.ObjVal)
