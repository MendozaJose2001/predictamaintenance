#src/repository/__init__.py
"""R repository package for PredictaMaintenance.

This package provides a clean Python interface to R survival analysis libraries,
specifically frailtypack for shared frailty Cox and AFT models. All rpy2
initialization and R package loading is handled here, ensuring that R is
started exactly once and that dependent modules can import R packages without
managing the rpy2 lifecycle themselves.

Data conversion strategy:
    pandas2ri is intentionally avoided in this package. Benchmarking revealed
    that pandas2ri produces subtle type differences that cause convergence
    failures. Instead, data is exchanged with R via temporary CSV files loaded
    with read.csv(), which replicates the native R type system.
    All R function calls use ro.r() string evaluation for the same reason.

Modules:
    frailty_cox: Wrapper for Cox proportional hazards with shared frailty.
    frailty_aft: Wrapper for Accelerated Failure Time with shared frailty.
        (pending implementation)
"""

import rpy2.rinterface_lib.callbacks as _cb
import rpy2.robjects as ro
from rpy2.robjects import packages

# ---------------------------------------------------------------------------
# R console output filtering
# ---------------------------------------------------------------------------

# R emits noisy "libraries contain no packages" warnings on every import
# when certain library paths are empty. These are environment warnings
# unrelated to model fitting and are suppressed here. All other R output
# (convergence warnings, errors, model diagnostics) is passed through.
_NOISE_PATTERNS: list[str] = [
    "libraries '/usr/local/lib/R/site-library'",
    "libraries '/usr/lib/R/site-library'",
    "contain no packages",
]


def _filter_r_console(s: str) -> None:
    """Passes through R output unless it matches a known noise pattern."""
    if not any(pattern in s for pattern in _NOISE_PATTERNS):
        print(s, end='')


_cb.consolewrite_print = _filter_r_console
_cb.consolewrite_warnerror = _filter_r_console

# ---------------------------------------------------------------------------
# R package loading
# ---------------------------------------------------------------------------

# Load R packages once at import time to avoid repeated loading overhead
_frailtypack = packages.importr('frailtypack')
_survival = packages.importr('survival')

__all__ = ['ro', 'packages', '_frailtypack', '_survival']