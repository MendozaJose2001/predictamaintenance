#./src/utils/ggs_io.py

"""I/O support for Group Grid Search — session management and persistence.

This module handles all file I/O for GGSTrainingManager, keeping persistence
logic completely separate from the training orchestration logic.

Directory structure managed by this module:

    outputs/ggs/
        metadata/
            {ModelName}_{hash}_{timestamp}.json   ← param_grid + run info
        checkpoints/
            {ModelName}_{hash}_{timestamp}.csv    ← partial results (in-progress)
        results/
            {ModelName}_{hash}_{timestamp}.csv    ← final results (completed)

Session identification:
    Each GGS run is identified by (model_name, param_grid_hash). The hash
    is a short MD5 of the serialized param_grid — same param_grid always
    produces the same hash, different param_grid produces a different hash.
    This ensures that changing the param_grid automatically creates a new
    session rather than conflicting with an existing one.

Resume logic:
    A checkpoint is detected when a file exists in outputs/ggs/checkpoints/
    matching the session's (model_name, param_grid_hash). If found, the
    session inherits the timestamp from that checkpoint and loads the
    previously completed configurations. The GGS then skips those configs
    and continues from where it left off.

Config identification:
    Each hyperparameter configuration is identified by a deterministic
    string key — the sorted serialization of its param dict. This key
    is stored in the checkpoint and used to detect which configs have
    already been evaluated.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_BASE_DIR = Path('outputs/ggs')

_METADATA_DIR    = 'metadata'
_CHECKPOINT_DIR  = 'checkpoints'
_RESULTS_DIR     = 'results'


def _ensure_dirs(base_dir: Path) -> None:
    """Creates the three subdirectories under base_dir if they don't exist."""
    for subdir in [_METADATA_DIR, _CHECKPOINT_DIR, _RESULTS_DIR]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hashing and key utilities
# ---------------------------------------------------------------------------

def compute_param_grid_hash(param_grid: dict) -> str:
    """Computes a short MD5 hash of the param_grid for session identification.

    The hash is deterministic — same param_grid always produces the same
    8-character hex string. Different param_grids produce different hashes,
    ensuring that changing hyperparameter ranges creates a new session.

    Args:
        param_grid: Dictionary mapping parameter names to candidate lists.

    Returns:
        8-character lowercase hex string (e.g. 'a3f2b1c4').
    """
    serialized = json.dumps(
        {k: sorted(str(v) for v in vs) for k, vs in sorted(param_grid.items())}
    )
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def config_key(params: dict) -> str:
    """Computes a deterministic string key for a single hyperparameter config.

    Used to identify which configurations have already been evaluated in
    a previous run. The key is the sorted serialization of the param dict.

    Args:
        params: Flat dictionary of hyperparameter values for one config.

    Returns:
        String of the form "[('key1', val1), ('key2', val2), ...]".
    """
    return str(sorted((k, str(v)) for k, v in params.items()))


# ---------------------------------------------------------------------------
# GGSSession dataclass
# ---------------------------------------------------------------------------

@dataclass
class GGSSession:
    """Encapsulates all state for a single GGS run.

    Created by resolve_ggs_session() — do not instantiate directly.

    Attributes:
        model_name:      Model class name (e.g. 'NegativeBinomialPiecewise').
        param_hash:      8-char MD5 hash of the param_grid.
        timestamp:       Run timestamp string 'YYYYMMDD_HHMM'.
        is_resume:       True if resuming from an existing checkpoint.
        path_metadata:   Path to the metadata JSON file.
        path_checkpoint: Path to the checkpoint CSV file.
        path_results:    Path to the final results CSV file.
        completed_keys:  Set of config_key() strings already evaluated.
        prior_results:   List of result dicts from the previous run.
    """
    model_name:      str
    param_hash:      str
    timestamp:       str
    is_resume:       bool
    path_metadata:   Path
    path_checkpoint: Path
    path_results:    Path
    completed_keys:  set[str]   = field(default_factory=set)
    prior_results:   list[dict] = field(default_factory=list)

    @property
    def stem(self) -> str:
        """Base filename stem: '{ModelName}_{hash}_{timestamp}'."""
        return f'{self.model_name}_{self.param_hash}_{self.timestamp}'


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------

def resolve_ggs_session(
    model_class: type,
    param_grid: dict,
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> GGSSession:
    """Resolves or creates a GGS session for the given model and param_grid.

    Detects whether a checkpoint exists for this (model, param_grid) pair.
    If found, returns a session configured for resumption with the inherited
    timestamp and previously completed configurations. If not found, returns
    a fresh session with a new timestamp.

    Args:
        model_class: Model class (not instance). Name extracted via __name__.
        param_grid:  Hyperparameter grid dict passed to group_grid_search().
        base_dir:    Root directory for GGS outputs. Defaults to outputs/ggs/.

    Returns:
        GGSSession configured for either a new run or resumption.
    """
    _ensure_dirs(base_dir)

    model_name = model_class.__name__
    param_hash = compute_param_grid_hash(param_grid)
    checkpoint_dir = base_dir / _CHECKPOINT_DIR

    # Search for existing checkpoint matching (model_name, param_hash)
    existing = sorted(checkpoint_dir.glob(f'{model_name}_{param_hash}_*.csv'))

    if existing:
        # Resume — inherit timestamp from most recent checkpoint
        checkpoint_path = existing[-1]

        # Extract timestamp robustly using param_hash as anchor.
        # stem = '{model_name}_{param_hash}_{timestamp}'
        # param_hash is always 8 hex chars — use it to find timestamp start.
        stem = checkpoint_path.stem
        hash_idx = stem.index(param_hash)
        timestamp = stem[hash_idx + len(param_hash) + 1:]  # after '{hash}_'

        # stem already contains model_name + param_hash + timestamp
        path_metadata   = base_dir / _METADATA_DIR   / f'{stem}.json'
        path_checkpoint = base_dir / _CHECKPOINT_DIR / f'{stem}.csv'
        path_results    = base_dir / _RESULTS_DIR    / f'{stem}.csv'

        checkpoint_df = pd.read_csv(path_checkpoint)
        prior_results = checkpoint_df.to_dict(orient='records')
        completed_keys = {
            config_key(_extract_params(r, param_grid))
            for r in prior_results
        }

        print(f"  ♻️  Resuming session: {checkpoint_path.name}")
        print(f"     Configs completed: {len(completed_keys)}")

        return GGSSession(
            model_name=model_name,
            param_hash=param_hash,
            timestamp=timestamp,
            is_resume=True,
            path_metadata=path_metadata,
            path_checkpoint=path_checkpoint,
            path_results=path_results,
            completed_keys=completed_keys,
            prior_results=prior_results,
        )

    else:
        # New session — generate fresh timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        stem = f'{model_name}_{param_hash}_{timestamp}'
        path_metadata   = base_dir / _METADATA_DIR   / f'{stem}.json'
        path_checkpoint = base_dir / _CHECKPOINT_DIR / f'{stem}.csv'
        path_results    = base_dir / _RESULTS_DIR    / f'{stem}.csv'

        print(f"  🆕 New GGS session: {stem}")

        return GGSSession(
            model_name=model_name,
            param_hash=param_hash,
            timestamp=timestamp,
            is_resume=False,
            path_metadata=path_metadata,
            path_checkpoint=path_checkpoint,
            path_results=path_results,
        )


def _extract_params(result: dict, param_grid: dict) -> dict:
    """Extracts only the hyperparameter keys from a result dict.

    Used to reconstruct the config_key from a loaded checkpoint row.

    Args:
        result:     A single result dict (one row from checkpoint CSV).
        param_grid: The param_grid defining which keys are hyperparameters.

    Returns:
        Dict with only the hyperparameter key-value pairs.
    """
    return {k: result[k] for k in param_grid if k in result}


# ---------------------------------------------------------------------------
# I/O operations
# ---------------------------------------------------------------------------

def save_metadata(
    session: GGSSession,
    param_grid: dict,
    n_folds: int,
    total_configs: int,
) -> None:
    """Writes the session metadata JSON file.

    Called once at the start of a new GGS run. Not called on resume
    since the metadata file already exists.

    Args:
        session:       Active GGSSession.
        param_grid:    Hyperparameter grid for this run.
        n_folds:       Number of GroupKFold folds.
        total_configs: Total number of configurations to evaluate.
    """
    metadata = {
        'model':         session.model_name,
        'param_hash':    session.param_hash,
        'timestamp':     session.timestamp,
        'n_folds':       n_folds,
        'total_configs': total_configs,
        'param_grid':    {k: list(v) for k, v in param_grid.items()},
    }
    with open(session.path_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_checkpoint(
    session: GGSSession,
    results: list[dict],
) -> None:
    """Overwrites the checkpoint CSV with all results evaluated so far.

    Called after every checkpoint_every completed configurations.
    Combines prior_results (from previous run) with new results.

    Args:
        session: Active GGSSession.
        results: List of result dicts evaluated in the current run.
    """
    all_results = session.prior_results + results
    pd.DataFrame(all_results).to_csv(session.path_checkpoint, index=False)


def save_results(
    session: GGSSession,
    results: list[dict],
) -> None:
    """Writes the final results CSV and removes the checkpoint file.

    Called once when the GGS completes all configurations. Combines
    prior_results with new results for the complete record.

    Args:
        session: Active GGSSession.
        results: List of all result dicts from the current run.
    """
    all_results = session.prior_results + results
    pd.DataFrame(all_results).to_csv(session.path_results, index=False)

    # Remove checkpoint — no longer needed
    if session.path_checkpoint.exists():
        session.path_checkpoint.unlink()
        print(f"Checkpoint removed: {session.path_checkpoint.name}")

    print(f"Results saved: {session.path_results}")


def delete_checkpoint(session: GGSSession) -> None:
    """Removes the checkpoint file if it exists.

    Called on successful completion or manual cleanup.

    Args:
        session: Active GGSSession.
    """
    if session.path_checkpoint.exists():
        session.path_checkpoint.unlink()