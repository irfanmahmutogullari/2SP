<p align="center"><img src="assets/logo.svg" width="320" alt="2SP"></p>

**2SP** builds two-stage stochastic (mixed-)integer programs as Gurobi models. You describe the problem with
NumPy or SciPy matrices; 2SP checks the data and builds the extensive form for you. It can also compute
the expected value of perfect information (EVPI) and the value of the stochastic solution (VSS), and read SMPS files such as the SIPLIB instances.

> **Status:** 0.1.0, an early release. The API may still change.

## The model

$$
\begin{aligned}
\min/\max\quad & c^\top x + \sum_{k} p_k  q_k^\top y_k \\
\text{s.t.}\quad & A x \; (\le, = , \ge) \;  b \\
& W y_k \; (\le, = , \ge) \; h_k - T_k x && \text{for every scenario } k \\
& l_1 \le x \le u_1, \quad l_2 \le y_k \le u_2
\end{aligned}
$$

- The recourse matrix $W$ is the same in every scenario; $q_k$, $h_k$ and $T_k$ may differ.
- The **last** `p1` entries of $x$ and the **last** `p2` entries of $y_k$ are integer.
- Each row can be `"<="`, `">="` or `"=="`. Default bounds are $0 \le x, y \le \infty$.
- Scenario probabilities are scaled to sum to 1, so `prob=1` for every scenario means "equally likely".

## Installation

2SP needs Python 3.9+ and [Gurobi](https://www.gurobi.com) 12 (`gurobipy`). It is not on PyPI yet;
install it from GitHub:

```bash
pip install git+https://github.com/irfanmahmutogullari/2SP
```

or, to work on the code and run the tests, from a clone:

```bash
git clone https://github.com/irfanmahmutogullari/2SP
cd 2SP
pip install -e ".[test]"
```

`pip install gurobipy` comes with a size-limited licence (2,000 variables and 2,000 constraints). That is enough
for the examples in [`examples/`](examples/). Larger models, such as the SSLP instances used in the tests, need a
full licence; Gurobi offers [free academic licences](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Quick start

An integer newsvendor: order `x` units at cost 1 before the demand (2 or 4, equally likely) is known.
Afterwards, a shortage costs 3 per unit and a surplus 0.5 per unit.

```python
from twosp import TwoStageModel, evpi, vss

sp = TwoStageModel(c=[1.0], p1=1)                 # first stage: x, integer
sp.set_second_stage(W=[[1.0, -1.0]])              # y = (shortage, surplus):  shortage - surplus = d - x

for demand in (2.0, 4.0):
    sp.add_scenario(prob=1, q=[3.0, 0.5], h=[demand], T=[[1.0]])

model = sp.build()                                # a gurobipy.Model
model.optimize()
print(sp.x.X, model.ObjVal)                       # [4.] 4.5
print(evpi(sp), vss(sp))                          # 1.5 0.25
```

## API

| | |
|---|---|
| `TwoStageModel(c, A=None, b=None, p1=0, lb=0, ub=inf, sense="==", objective="min")` | First stage. |
| `sp.set_second_stage(W, p2=0, lb=0, ub=inf, sense="==")` | Second stage. Cannot be changed once scenarios have been added. |
| `sp.add_scenario(prob, q, h, T)` | Adds one scenario. |
| `sp.build()` | Returns the extensive form as a `gurobipy.Model`; the variables are `sp.x` and `sp.ys[k]` (Gurobi `MVar`s). |
| `recourse_problem`, `wait_and_see`, `expected_value_problem`, `expected_result_of_ev`, `evpi`, `vss` | RP, WS, EV, EEV, EVPI and VSS as in Birge & Louveaux, Ch. 4. EVPI and VSS are always $\ge 0$, also for `"max"`. |
| `read_smps(cor, tim, sto)` | Reads SMPS files; returns `(sp, x_names, y_names)`. |

Vectors and matrices may be lists, NumPy arrays or SciPy sparse matrices. Wrong input raises an error that
names the argument, e.g. `ValueError: scenario 3: h must have shape (2,), got (3,)`.

`read_smps` supports two-stage `SCENARIOS DISCRETE` files whose scenarios all have parent `ROOT`, with random
$q$, $h$ and $T$ (not $W$). Within each stage it reorders the columns so that the integer ones come last;
`x_names` and `y_names` give the new order.

## Examples

| File | Problem |
|---|---|
| [`newsvendor.py`](examples/newsvendor.py) | The integer newsvendor above. |
| [`farmer.py`](examples/farmer.py) | The farmer's problem (Birge & Louveaux, Sec. 1.1), with EVPI and VSS. |
| [`example1_multiknapsack.py`](examples/example1_multiknapsack.py) | A stochastic multi-knapsack problem with 441 scenarios (Carøe & Schultz, Example 1). |
| [`siplib.py`](examples/siplib.py) | Reads and solves the SIPLIB instance SSLP_5_25_50 (needs a full Gurobi licence). |

## Tests

```bash
pytest
```

The tests compare against published optimal values: the farmer's problem, the multi-knapsack problem, and
the SIPLIB instances SIZES and SSLP. The SSLP tests need a full Gurobi licence.

## References

- J. R. Birge and F. Louveaux. *Introduction to Stochastic Programming*, 2nd ed. Springer, 2011.
- C. C. Carøe and R. Schultz. Dual decomposition in stochastic integer programming. *Operations Research Letters* 24(1–2):37–45, 1999. (ZIB preprint SC 96-46, 1996.)
- S. Ahmed. Two-stage stochastic integer programming: a brief introduction. In *Wiley Encyclopedia of Operations Research and Management Science*, 2010.
- S. Ahmed, R. Garcia, N. Kong, L. Ntaimo, G. Parija, F. Qiu and S. Sen. SIPLIB: a stochastic integer programming test problem library. <https://www2.isye.gatech.edu/~sahmed/siplib>. The instances in [`tests/data`](tests/data) come from SIPLIB.

## Acknowledgements

Parts of the code, tests and documentation were developed with the help of
[Claude Code](https://claude.com/claude-code) (Anthropic). All code was reviewed and tested by the author.


## Licence

[MIT](LICENSE)
