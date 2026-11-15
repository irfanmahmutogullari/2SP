"""Integer newsvendor: order x units (cost 1 each) before demand d is known.
Second stage: shortage s (cost 3) and surplus u (cost 0.5), with s - u = d - x.
"""
import numpy as np
from twosp import TwoStageModel

sp = TwoStageModel(c=np.array([1.0]), p1=1)       # first stage: x integer
sp.set_second_stage(W=np.array([[1.0, -1.0]]))    # y = (s, u)

q, T = np.array([3.0, 0.5]), np.array([[1.0]])
sp.add_scenario(prob=1, q=q, h=np.array([2.0]), T=T)  # low demand
sp.add_scenario(prob=1, q=q, h=np.array([4.0]), T=T)  # high demand (weights scaled to 0.5/0.5)

model = sp.build()
model.optimize()
print("x* =", sp.x.X, " objective =", model.ObjVal)
