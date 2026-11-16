"""Global test setup: one native thread per library, as in production (see riskml.server).

threadpoolctl can only limit libraries that are already loaded, so the environment variables
are set first (read when OpenMP/BLAS initialise) and the limit is applied after importing
scikit-learn's compiled modules.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sklearn.ensemble  # noqa: E402, F401  (loads the OpenMP runtime)
from threadpoolctl import threadpool_limits  # noqa: E402

_LIMITS = threadpool_limits(limits=1)
