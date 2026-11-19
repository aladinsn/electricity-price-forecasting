"""One-off check of `src.evaluation.diebold_mariano` against `epftoolbox.evaluation.DM` (multivariate).

Not collected by pytest (no `test_` prefix): epftoolbox is not a project dependency, since its install
requirements pull TensorFlow and Keras. `epftoolbox.evaluation` itself only imports numpy, scipy, pandas
and matplotlib, so run this script in a throwaway environment outside the project, e.g.:

    uv venv --python 3.12 /tmp/epf-check
    uv pip install --python /tmp/epf-check numpy scipy pandas matplotlib
    uv pip install --python /tmp/epf-check --no-deps "epftoolbox @ git+https://github.com/jeslago/epftoolbox.git"
    /tmp/epf-check/bin/python tests/check_dm_against_epftoolbox.py

The result is recorded in the docstring of `diebold_mariano`.
"""

import sys
from pathlib import Path

import numpy as np
import scipy
from epftoolbox.evaluation import DM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import diebold_mariano  # noqa: E402

N_CASES = 2000
rng = np.random.default_rng(0)
max_diff, p_values = 0.0, []
for i in range(N_CASES):
    n_days = int(rng.integers(5, 401))
    norm = 1 + i % 2
    scale = rng.uniform(1, 50)
    # Prices with negative values and spikes; forecast B sometimes biased, sometimes better than A
    y = rng.standard_t(3, size=(n_days, 24)) * scale + 80
    pred_a = y + rng.standard_t(3, size=(n_days, 24)) * scale
    pred_b = y + rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 1.5) * scale, size=(n_days, 24))

    p_ref = DM(p_real=y, p_pred_1=pred_a, p_pred_2=pred_b, norm=norm, version="multivariate")
    p_ours = diebold_mariano(y - pred_a, y - pred_b, norm=norm)
    max_diff = max(max_diff, abs(p_ref - p_ours))
    p_values.append(p_ours)

p_values = np.array(p_values)
print(f"numpy {np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}")
print(f"{N_CASES} cases, p-values from {p_values.min():.2e} to {p_values.max():.3f}, "
      f"{np.mean(p_values < 0.05):.0%} below 0.05")
print(f"max |p_epftoolbox - p_ours| = {max_diff:.2e}")
assert max_diff < 1e-12
