"""Read a SIPLIB instance in SMPS format and solve its extensive form.
SIPLIB: https://www2.isye.gatech.edu/~sahmed/siplib  (SSLP_5_25_50 optimum: -121.60)
"""
from pathlib import Path

from twosp import read_smps

data = Path(__file__).parent.parent / "tests" / "data"
sp, x_names, y_names = read_smps(data / "sslp_5_25_50.cor", data / "sslp_5_25_50.tim", data / "sslp_5_25_50.sto")

model = sp.build()
model.optimize()
print("objective =", model.ObjVal)
print("open servers:", [n for n, v in zip(x_names, sp.x.X) if v > 0.5])
