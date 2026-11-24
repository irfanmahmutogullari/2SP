"""Block structure of the extensive form (needs matplotlib):

    [ A                 ]   first stage
    [ T_1  W            ]   scenario 1
    [ T_2      W        ]   scenario 2
    [ ...          ...  ]
"""
import numpy as np
from scipy import sparse


def extensive_matrix(sp):
    """Constraint matrix of the extensive form, as a SciPy sparse matrix."""
    K = len(sp.scenarios)
    A = sparse.csr_matrix(sp.A if sp.A is not None else (0, len(sp.c)))
    blocks = [[A] + [None] * K]
    for k, (_, _, _, T) in enumerate(sp.scenarios):
        blocks.append([sparse.csr_matrix(T)] + [sparse.csr_matrix(sp.W) if j == k else None for j in range(K)])
    return sparse.bmat(blocks, format="csr")


def plot_structure(sp, ax=None, markersize=None):
    """Draws the nonzeros of the extensive form, with lines between the blocks; returns the axes."""
    import matplotlib.pyplot as plt

    M = extensive_matrix(sp)
    if ax is None:
        _, ax = plt.subplots()
    if markersize is None:
        markersize = float(np.clip(300 / max(M.shape), 0.5, 5))
    ax.spy(M, markersize=markersize, aspect="auto")

    K = len(sp.scenarios)
    m1 = 0 if sp.A is None else sp.A.shape[0]
    m2, n2 = sp.W.shape if K else (0, 0)
    rows = [0] + [m1 + k * m2 for k in range(K + 1)]  # block borders
    cols = [0] + [len(sp.c) + k * n2 for k in range(K + 1)]
    if K <= 20:  # with more scenarios the lines and labels hide the structure
        for r in rows[1:-1]:
            ax.axhline(r - 0.5, color="gray", lw=0.5)
        for c in cols[1:-1]:
            ax.axvline(c - 0.5, color="gray", lw=0.5)
        ax.set_xticks([(a + b) / 2 - 0.5 for a, b in zip(cols, cols[1:])],
                      ["x"] + [f"$y_{{{k + 1}}}$" for k in range(K)])
        ax.set_yticks([(a + b) / 2 - 0.5 for a, b in zip(rows, rows[1:])],
                      ["first stage"] + [f"scenario {k + 1}" for k in range(K)])
        ax.tick_params(length=0)
    ax.set_title(f"{M.shape[0]} rows, {M.shape[1]} columns, {M.nnz} nonzeros")
    return ax
