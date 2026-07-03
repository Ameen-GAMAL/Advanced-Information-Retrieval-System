"""Fusion-weight tuning framed as a constrained optimisation problem.

We search for weights ``w`` on the probability simplex
(``w_r >= 0``, ``sum_r w_r = 1``) that maximise MAP of the fused ranking
on a held-out validation query set:

    w* = argmax_w  MAP( fuse(runs, w), qrels_val )

MAP is a piecewise-constant, non-differentiable function of the weights
(it only changes when two documents swap rank positions), so
gradient-based optimisers are unsuitable. Instead we combine:

1. **Coarse grid search** over the simplex to find a good basin, then
2. **Nelder-Mead simplex refinement** in an unconstrained space obtained
   by re-parameterising the weights with a softmax, which keeps every
   candidate on the simplex without explicit constraints.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Mapping

import numpy as np
import pandas as pd

from .evaluation import mean_average_precision
from .fusion import fuse

logger = logging.getLogger(__name__)

MetricFn = Callable[[pd.DataFrame, pd.DataFrame], float]


def softmax(theta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax mapping R^n onto the open simplex."""
    shifted = theta - np.max(theta)
    exp = np.exp(shifted)
    return exp / exp.sum()


@dataclass
class OptimizationResult:
    """Outcome of a weight-tuning run."""

    weights: Dict[str, float]
    validation_score: float
    baseline_score: float
    history: list

    @property
    def relative_improvement(self) -> float:
        """Relative gain of the tuned fusion over the baseline ranker."""
        if self.baseline_score == 0:
            return 0.0
        return (self.validation_score - self.baseline_score) / self.baseline_score


class WeightOptimizer:
    """Tunes late-fusion weights to maximise a rank metric (MAP by default).

    Parameters
    ----------
    runs:
        Mapping from ranker name to its run on the *validation* queries.
        Runs are normalised once up front so each objective evaluation
        only pays for the weighted sum and the metric.
    qrels:
        Validation qrels used to compute the objective.
    baseline:
        Name of the ranker to report improvement against (e.g. "bm25").
    metric:
        Metric function ``f(run, qrels) -> float`` to maximise.
    """

    def __init__(
        self,
        runs: Mapping[str, pd.DataFrame],
        qrels: pd.DataFrame,
        baseline: str = "bm25",
        metric: MetricFn = mean_average_precision,
    ) -> None:
        if baseline not in runs:
            raise KeyError(f"baseline ranker '{baseline}' not among runs {sorted(runs)}")
        self.names = list(runs)
        self.runs = dict(runs)
        self.qrels = qrels
        self.baseline = baseline
        self.metric = metric
        self.history: list = []

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #

    def objective(self, weights: np.ndarray) -> float:
        """Metric value of the fusion under ``weights`` (higher is better)."""
        mapping = {name: float(w) for name, w in zip(self.names, weights)}
        fused = fuse(self.runs, mapping)
        value = self.metric(fused, self.qrels)
        self.history.append((mapping, value))
        return value

    # ------------------------------------------------------------------ #
    # Search strategies
    # ------------------------------------------------------------------ #

    def grid_search(self, resolution: int = 5) -> np.ndarray:
        """Exhaustive search over a lattice on the simplex.

        Enumerates all weight vectors whose components are multiples of
        ``1/(resolution-1)`` and sum to 1, evaluating the objective for
        each. Returns the best weight vector found.
        """
        levels = np.linspace(0.0, 1.0, resolution)
        best_w: np.ndarray | None = None
        best_value = -np.inf
        seen = set()
        for combo in itertools.product(levels, repeat=len(self.names)):
            total = sum(combo)
            if total == 0:
                continue
            w = np.asarray(combo) / total
            key = tuple(np.round(w, 6))
            if key in seen:
                continue
            seen.add(key)
            value = self.objective(w)
            if value > best_value:
                best_value, best_w = value, w
        logger.info("grid search best %s = %.4f at %s", self.metric.__name__, best_value, best_w)
        return best_w

    def refine(self, w0: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Nelder-Mead refinement around ``w0`` via softmax re-parameterisation.

        The simplex constraint is folded into the parameterisation
        ``w = softmax(theta)``, turning the problem into an unconstrained
        search over ``theta`` that Nelder-Mead handles well despite the
        piecewise-constant objective.
        """
        from scipy.optimize import minimize

        # invert the softmax at w0 (up to an additive constant)
        theta0 = np.log(np.clip(w0, 1e-6, None))
        result = minimize(
            lambda theta: -self.objective(softmax(theta)),
            theta0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-6},
        )
        return softmax(result.x)

    def optimize(self, grid_resolution: int = 5, max_iter: int = 200) -> OptimizationResult:
        """Full tuning pipeline: grid search, then local refinement."""
        coarse = self.grid_search(resolution=grid_resolution)
        refined = self.refine(coarse, max_iter=max_iter)

        candidates = [coarse, refined]
        values = [self.objective(w) for w in candidates]
        best = candidates[int(np.argmax(values))]
        best_value = max(values)

        baseline_value = self.metric(self.runs[self.baseline], self.qrels)
        weights = {name: float(w) for name, w in zip(self.names, best)}
        logger.info(
            "tuned weights %s -> validation %s %.4f (baseline %.4f, %+.1f%%)",
            weights,
            self.metric.__name__,
            best_value,
            baseline_value,
            100 * (best_value - baseline_value) / max(baseline_value, 1e-9),
        )
        return OptimizationResult(
            weights=weights,
            validation_score=best_value,
            baseline_score=baseline_value,
            history=self.history,
        )
