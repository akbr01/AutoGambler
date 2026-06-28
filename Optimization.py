#!/bin/python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import random


@dataclass
class UniformFloat:
    low: float
    high: float


@dataclass
class UniformInt:
    low: int
    high: int


@dataclass
class Choice:
    options: list


Param = UniformFloat | UniformInt | Choice | int | float | str | bool


def _sample_param(param: Param) -> Any:
    if isinstance(param, UniformFloat):
        return random.uniform(param.low, param.high)
    elif isinstance(param, UniformInt):
        return random.randint(param.low, param.high)
    elif isinstance(param, Choice):
        return random.choice(param.options)
    else:
        return param


class RandomSearch:
    def __init__(
        self,
        param_space: dict[str, Param],
        eval_fn: Callable[[dict[str, Any]], float],
        n_trials: int,
        maximize: bool = True,
        seed: int | None = None,
    ):
        self.param_space = param_space
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        self.maximize = maximize
        if seed is not None:
            random.seed(seed)

    def search(self, plot: bool = False) -> list[tuple[dict[str, Any], float]]:
        results: list[tuple[dict[str, Any], float]] = []
        best_score = float("-inf") if self.maximize else float("inf")
        best_params = None

        for i in range(self.n_trials):
            params = {k: _sample_param(v) for k, v in self.param_space.items()}
            score = self.eval_fn(params)
            results.append((params, score))

            better = score > best_score if self.maximize else score < best_score
            if better:
                best_score = score
                best_params = params

            print(f"Trial {i+1}/{self.n_trials}: score={score:.4f}  best={best_score:.4f}")

        print(f"\nBest: {best_params} -> {best_score:.4f}")

        if plot:
            self.plot_results(results)

        return results

    def plot_results(self, results: list[tuple[dict[str, Any], float]]):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        rows = [{**p, "score": s} for p, s in results]
        df = pd.DataFrame(rows)

        numeric_cols = [k for k, v in self.param_space.items() if isinstance(v, (UniformFloat, UniformInt))]
        cat_cols = [k for k, v in self.param_space.items() if isinstance(v, Choice)]

        n_plots = len(numeric_cols) + len(cat_cols)
        if n_plots == 0:
            print("No plottable parameters")
            return

        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            sns.regplot(data=df, x=col, y="score", ax=axes[i], scatter_kws={"alpha": 0.5, "s": 20}, line_kws={"color": "red"}, lowess=True)
            axes[i].set_title(col)

        for i, col in enumerate(cat_cols, start=len(numeric_cols)):
            sns.stripplot(data=df, x=col, y="score", ax=axes[i], alpha=0.5)
            axes[i].set_title(col)

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()
