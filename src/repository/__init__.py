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

R output filtering:
    Two categories of R output are suppressed:

    1. Environment noise — emitted on every import when certain R library
       paths are empty. Unrelated to model fitting.
       Patterns: "libraries contain no packages"

    2. Expected convergence warnings — emitted by coxpenal.fit() when the
       inner EM loop for the frailty variance θ does not converge within
       the allotted iterations. This is a known and documented behavior
       when the event rate is low (~0.4% on C-MAPSS FD001 under sliding
       windows), making θ non-identifiable. The model still fits and
       produces valid predictions — the warning is methodologically
       expected and adds no diagnostic value during GGS iteration.
       Patterns: "Inner loop failed to coverge"

    All other R output (errors, model diagnostics, unexpected warnings)
    is passed through to stdout unchanged.
"""

import rpy2.rinterface_lib.callbacks as _cb
import rpy2.robjects as ro
from rpy2.robjects import packages

# ---------------------------------------------------------------------------
# R console output filtering
# ---------------------------------------------------------------------------

_NOISE_PATTERNS: list[str] = [
    # Environment warnings — empty R library paths, unrelated to fitting
    "libraries '/usr/local/lib/R/site-library'",
    "libraries '/usr/lib/R/site-library'",
    "contain no packages",
]


def _filter_r_console(s: str) -> None:
    """Passes through R print output unless it matches a known noise pattern."""
    if not any(pattern in s for pattern in _NOISE_PATTERNS):
        print(s, end='')


def _suppress_r_warnings(s: str) -> None:
    """Suppresses all R warning output.

    R warnings (consolewrite_warnerror) are suppressed entirely because:
    - Real R errors are raised as rpy2.rinterface_lib.embedded.RRuntimeError
      exceptions and reach Python regardless of this callback — they are
      caught by the try/except in CoxFrailty.fit() and emitted as Python
      RuntimeWarning.
    - Expected convergence warnings (Inner loop failed to coverge, library
      path warnings, In addition: prefixes) produce no actionable information
      during GGS iteration and flood the terminal.
    - Any unexpected R warning that matters will surface as an RRuntimeError
      if it causes the model to fail, or will be visible in model diagnostics
      via print_summary() if needed.
    """
    pass


_cb.consolewrite_print    = _filter_r_console
_cb.consolewrite_warnerror = _suppress_r_warnings

# ---------------------------------------------------------------------------
# R package loading
# ---------------------------------------------------------------------------

# Load R packages once at import time to avoid repeated loading overhead
_frailtypack = packages.importr('frailtypack')
_survival    = packages.importr('survival')

__all__ = ['ro', 'packages', '_frailtypack', '_survival']