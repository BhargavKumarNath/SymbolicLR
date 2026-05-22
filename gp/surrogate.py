"""
gp/surrogate.py - Lightweight online surrogate fitness predictor.

Uses Ridge Regression on a 10-feature schedule/tree descriptor vector to
predict fitness. Acts as a triage layer to reduce expensive GPU evaluations.

Safety constraints (non-negotiable):
- Filters at most 30% of candidates (eval_fraction=0.70)
- Structurally new candidates (unseen hash) are ALWAYS evaluated
- High-novelty candidates are ALWAYS evaluated
- Surrogate only activates after min_samples real evaluations
- Never replaces real evaluation — only prioritizes it

Independently disable-able via config.surrogate_enabled.
No GPU dependency. CPU-only Ridge Regression (sklearn).

Features (10, pure numpy — zero extra cost):
    tree_size, tree_depth, mean_lr, std_lr, min_lr, max_lr,
    center_of_mass, total_variation, start_lr, end_lr
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from gp.tree import Node
from gp.rust_bridge import evaluate_schedule


def _compute_features(tree: Node, t_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute 10 cheap schedule/tree features. Pure numpy, no GPU.

    All features are designed to be fast to compute (O(n) where n=time_steps=100).
    Returns None if schedule evaluation fails or produces non-finite values.
    """
    try:
        lr_schedule = evaluate_schedule(tree, t_array)
        if not np.all(np.isfinite(lr_schedule)) or len(lr_schedule) == 0:
            return None

        abs_lr = np.abs(lr_schedule)
        total_lr = float(np.sum(abs_lr))
        com = (
            float(np.sum(t_array * abs_lr) / total_lr) if total_lr > 1e-10 else 0.5
        )

        return np.array(
            [
                float(tree.size()),
                float(tree.depth()),
                float(np.mean(lr_schedule)),
                float(np.std(lr_schedule)),
                float(np.min(lr_schedule)),
                float(np.max(lr_schedule)),
                com,
                float(np.sum(np.abs(np.diff(lr_schedule)))),
                float(lr_schedule[0]),
                float(lr_schedule[-1]),
            ],
            dtype=np.float64,
        )
    except Exception:
        return None


class LightweightSurrogate:
    """
    Online Ridge Regression surrogate for fitness prediction and candidate triage.

    Maintains a rolling buffer of real evaluations and refits on each update.
    Used exclusively for candidate ranking and early rejection — never replaces
    real evaluation for structurally novel or high-novelty candidates.

    The surrogate is conservative by design:
    - Only filters 30% of candidates (eval_fraction=0.70 evaluates top 70%)
    - Only activates after min_samples=50 real evaluations
    - Falls back gracefully if sklearn is unavailable or fitting fails

    Args:
        min_samples:    Minimum real evaluations before surrogate activates.
        eval_fraction:  Fraction of triageable candidates to evaluate (top-ranked).
                        Default 0.70 means at most 30% are filtered.
        buffer_size:    Rolling buffer size for training data.
    """

    def __init__(
        self,
        min_samples: int = 50,
        eval_fraction: float = 0.70,
        buffer_size: int = 200,
    ):
        if not SKLEARN_AVAILABLE:
            warnings.warn(
                "scikit-learn not available. LightweightSurrogate will not activate.",
                RuntimeWarning,
            )

        self.min_samples = min_samples
        self.eval_fraction = eval_fraction
        self.buffer_size = buffer_size

        self._X: deque = deque(maxlen=buffer_size)
        self._y: deque = deque(maxlen=buffer_size)
        self._model: Optional["Ridge"] = None
        self._scaler: Optional["StandardScaler"] = None
        self._fitted: bool = False

        # Rolling prediction errors for RMSE tracking
        self._prediction_errors: deque = deque(maxlen=50)
        # False negative tracking: surrogate-rejected candidates that would have
        # entered archive. Tracked by periodically evaluating a random sample.
        self._false_negatives: int = 0
        self._false_negative_checks: int = 0

    def is_ready(self) -> bool:
        """
        True when the surrogate has enough real evaluations to make predictions.
        Safe default: False → all candidates go to real evaluation.
        """
        return SKLEARN_AVAILABLE and self._fitted and len(self._X) >= self.min_samples

    def update(self, tree: Node, real_fitness: float, t_array: np.ndarray) -> None:
        """
        Add a real evaluation to the training buffer and refit the surrogate.

        Call after every real GPU evaluation to keep the surrogate current.
        Only refits when >= min_samples are available (cheap no-op otherwise).

        Args:
            tree:         The evaluated tree.
            real_fitness: The real fitness value from the evaluator.
            t_array:      Time array for feature extraction.
        """
        if not SKLEARN_AVAILABLE or not np.isfinite(real_fitness):
            return

        features = _compute_features(tree, t_array)
        if features is None:
            return

        # Track prediction error if we had a prior prediction
        if self._fitted:
            try:
                predicted = self._predict_raw(features)
                if np.isfinite(predicted):
                    self._prediction_errors.append(abs(predicted - real_fitness))
            except Exception:
                pass

        self._X.append(features)
        self._y.append(real_fitness)

        # Refit when enough samples are available
        if len(self._X) >= self.min_samples:
            self._refit()

    def _refit(self) -> None:
        """Refit the Ridge Regression model on the current buffer. Fails silently."""
        try:
            X = np.array(self._X)
            y = np.array(self._y)

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._model = Ridge(alpha=1.0)
            self._model.fit(X_scaled, y)
            self._fitted = True
        except Exception:
            self._fitted = False

    def _predict_raw(self, features: np.ndarray) -> float:
        """Internal: predict fitness from feature vector. Returns inf on failure."""
        if not self.is_ready():
            return float("inf")
        try:
            X_scaled = self._scaler.transform(features.reshape(1, -1))
            return float(self._model.predict(X_scaled)[0])
        except Exception:
            return float("inf")

    def predict(self, tree: Node, t_array: np.ndarray) -> float:
        """
        Predict fitness for a tree. Returns float('inf') if not ready.

        Args:
            tree:    The candidate tree.
            t_array: Time array for feature extraction.
        """
        if not self.is_ready():
            return float("inf")
        features = _compute_features(tree, t_array)
        if features is None:
            return float("inf")
        return self._predict_raw(features)

    def rank_candidates(
        self,
        candidates: List[Node],
        t_array: np.ndarray,
        always_evaluate: Optional[List[bool]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Split candidates into 'evaluate' and 'skip' lists using surrogate predictions.

        Candidates marked always_evaluate=True are unconditionally in the evaluate list.
        Among the remaining candidates, the top eval_fraction (by predicted fitness)
        are evaluated; the rest are skipped.

        Args:
            candidates:       List of candidate trees.
            t_array:          Time array for feature extraction.
            always_evaluate:  Boolean mask. True = always evaluate regardless of surrogate.

        Returns:
            (evaluate_indices, skip_indices): Two lists of integer indices.
            All always_evaluate indices appear in evaluate_indices.
        """
        n = len(candidates)
        if n == 0:
            return [], []

        if not self.is_ready():
            return list(range(n)), []

        if always_evaluate is None:
            always_evaluate = [False] * n

        must_eval = [i for i, flag in enumerate(always_evaluate) if flag]
        can_triage = [i for i, flag in enumerate(always_evaluate) if not flag]

        if not can_triage:
            return list(range(n)), []

        # Predict fitness for triageable candidates
        predictions = []
        for i in can_triage:
            pred = self.predict(candidates[i], t_array)
            predictions.append((pred, i))

        predictions.sort(key=lambda x: x[0])  # best (lowest predicted loss) first

        n_triage_eval = max(1, int(len(can_triage) * self.eval_fraction))
        triage_eval = [i for _, i in predictions[:n_triage_eval]]
        triage_skip = [i for _, i in predictions[n_triage_eval:]]

        return must_eval + triage_eval, triage_skip

    def record_false_negative_check(self, would_have_entered_archive: bool) -> None:
        """
        Record whether a surrogate-skipped candidate would have entered the archive.
        Used to track false negative rate as a quality metric.
        """
        self._false_negative_checks += 1
        if would_have_entered_archive:
            self._false_negatives += 1

    def get_stats(self) -> dict:
        """Return surrogate performance stats for diagnostics logging."""
        rmse = -1.0
        if self._prediction_errors:
            rmse = round(
                float(np.sqrt(np.mean(np.array(self._prediction_errors) ** 2))), 4
            )

        fnr = (
            round(self._false_negatives / self._false_negative_checks, 3)
            if self._false_negative_checks > 0
            else -1.0
        )

        return {
            "is_ready": self.is_ready(),
            "buffer_size": len(self._X),
            "prediction_rmse": rmse,
            "false_negative_rate": fnr,
            "sklearn_available": SKLEARN_AVAILABLE,
        }
