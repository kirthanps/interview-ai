from __future__ import annotations

from itertools import combinations
from math import sqrt
from statistics import mean
from typing import Any


def _cohen_d(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)
    vy = sum((y - my) ** 2 for y in ys) / (len(ys) - 1)
    pooled = ((len(xs) - 1) * vx + (len(ys) - 1) * vy) / \
        (len(xs) + len(ys) - 2)
    if pooled <= 0:
        return 0.0
    return (mx - my) / sqrt(pooled)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sqrt(sum((x - mx) ** 2 for x in xs))
    deny = sqrt(sum((y - my) ** 2 for y in ys))
    den = denx * deny
    if den == 0:
        return 0.0
    return num / den


def one_way_anova(groups: dict[str, list[float]]) -> dict[str, Any]:
    keys = list(groups.keys())
    data = [groups[k] for k in keys]
    try:
        from scipy.stats import f_oneway  # type: ignore

        f_stat, p_value = f_oneway(*data)
        return {"f_stat": float(f_stat), "p_value": float(p_value), "method": "scipy"}
    except Exception:
        all_values = [v for g in data for v in g]
        if not all_values:
            return {"f_stat": 0.0, "p_value": None, "method": "manual_no_p"}

        overall_mean = mean(all_values)
        ss_between = sum(len(g) * (mean(g) - overall_mean)
                         ** 2 for g in data if g)
        ss_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in data if g)
        df_between = max(1, len(data) - 1)
        df_within = max(1, len(all_values) - len(data))
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within if df_within else 0.0
        f_stat = ms_between / ms_within if ms_within else 0.0
        return {"f_stat": float(f_stat), "p_value": None, "method": "manual_no_p"}


def pairwise_t_tests(groups: dict[str, list[float]]) -> list[dict[str, Any]]:
    results = []
    keys = list(groups.keys())

    for left, right in combinations(keys, 2):
        xs = groups[left]
        ys = groups[right]
        d = _cohen_d(xs, ys)

        p_value = None
        method = "manual_effect_only"
        try:
            from scipy.stats import ttest_ind  # type: ignore

            stat = ttest_ind(xs, ys, equal_var=False)
            p_value = float(stat.pvalue)
            method = "scipy"
        except Exception:
            pass

        results.append(
            {
                "left": left,
                "right": right,
                "p_value": p_value,
                "cohen_d": float(d),
                "method": method,
            }
        )

    return results


def correlation_matrix(series_by_model: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    aliases = list(series_by_model.keys())
    matrix: dict[str, dict[str, float]] = {}
    for a in aliases:
        row: dict[str, float] = {}
        for b in aliases:
            row[b] = float(_pearson(series_by_model.get(
                a, []), series_by_model.get(b, [])))
        matrix[a] = row
    return matrix
