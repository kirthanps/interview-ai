from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from typing import Any


def _tokenize(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()}


def _bounded(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def question_metrics(questions: list[dict[str, Any]], resume_context: str, jd_context: str) -> dict[str, float]:
    resume_tokens = _tokenize(resume_context)
    jd_tokens = _tokenize(jd_context)
    question_texts = [str(q.get("question_text") or q.get(
        "question") or "") for q in questions]
    hints = [q.get("hint", q.get("hints", "")) for q in questions]
    rationales = [q.get("rationale", "") for q in questions]

    if not questions:
        return {
            "resume_relevance": 0.0,
            "jd_alignment": 0.0,
            "diversity": 0.0,
            "hint_quality": 0.0,
            "contextualization": 0.0,
            "qms": 0.0,
        }

    resume_overlap_scores = []
    jd_overlap_scores = []
    context_scores = []

    for idx, text in enumerate(question_texts):
        qt = _tokenize(text)
        if resume_tokens:
            resume_overlap_scores.append(
                100.0 * len(qt & resume_tokens) / len(resume_tokens))
        else:
            resume_overlap_scores.append(0.0)

        if jd_tokens:
            jd_overlap_scores.append(
                100.0 * len(qt & jd_tokens) / len(jd_tokens))
        else:
            jd_overlap_scores.append(0.0)

        rt = _tokenize(rationales[idx] if idx < len(rationales) else "")
        combined = qt | rt
        ref_tokens = (resume_tokens | jd_tokens)
        context_scores.append(
            100.0 * len(combined & ref_tokens) / max(1, len(combined)))

    unique_ratio = len({q.strip().lower()
                       for q in question_texts}) / len(question_texts)
    diversity = 100.0 * unique_ratio

    hint_scores = []
    for hint in hints:
        if isinstance(hint, list):
            hint_text = " ".join(str(item) for item in hint)
        else:
            hint_text = str(hint)
        token_count = len(_tokenize(hint_text))
        hint_scores.append(100.0 if token_count >=
                           8 else (token_count / 8.0) * 100.0)

    resume_relevance = _bounded(mean(resume_overlap_scores) * 4.0)
    jd_alignment = _bounded(mean(jd_overlap_scores) * 4.0)
    hint_quality = _bounded(mean(hint_scores))
    contextualization = _bounded(mean(context_scores))

    qms = _bounded(
        0.25 * resume_relevance
        + 0.25 * jd_alignment
        + 0.20 * diversity
        + 0.15 * hint_quality
        + 0.15 * contextualization
    )

    return {
        "resume_relevance": round(resume_relevance, 4),
        "jd_alignment": round(jd_alignment, 4),
        "diversity": round(diversity, 4),
        "hint_quality": round(hint_quality, 4),
        "contextualization": round(contextualization, 4),
        "qms": round(qms, 4),
    }


def _avg_question_score(model_scoring_output: dict[str, Any]) -> float:
    rows = model_scoring_output.get("question_scores", [])
    if not rows:
        return 0.0
    vals = []
    for row in rows:
        vals.append(mean([float(row.get("relevance", 0)), float(
            row.get("depth", 0)), float(row.get("clarity", 0))]))
    return mean(vals)


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return num / den


def scoring_metrics(
    score_outputs_by_model: dict[str, dict[str, Any]],
    answer_payload: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Returns SMS metric bundle per evaluated model for one shared answer set.
    """
    model_aliases = list(score_outputs_by_model.keys())

    per_model_question_means: dict[str, list[float]] = defaultdict(list)
    per_model_skip_scores: dict[str, list[float]] = defaultdict(list)

    labels_by_qid = {
        item.get("question_id", ""): str(item.get("answer_label", "")).upper()
        for item in answer_payload.get("responses", [])
    }

    for alias, output in score_outputs_by_model.items():
        for row in output.get("question_scores", []):
            qid = row.get("question_id", "")
            q_mean = mean([float(row.get("relevance", 0)), float(
                row.get("depth", 0)), float(row.get("clarity", 0))])
            per_model_question_means[alias].append(q_mean)

            label = labels_by_qid.get(qid, "")
            if label == "SKIPPED":
                expected_max = 25.0
                score = max(0.0, 100.0 - max(0.0, q_mean - expected_max) * 4.0)
                per_model_skip_scores[alias].append(score)
            elif label == "SHORT":
                expected_max = 45.0
                score = max(0.0, 100.0 - max(0.0, q_mean - expected_max) * 2.0)
                per_model_skip_scores[alias].append(score)

    all_avg_scores = {alias: _avg_question_score(
        out) for alias, out in score_outputs_by_model.items()}
    global_mean = mean(all_avg_scores.values()) if all_avg_scores else 0.0

    result: dict[str, dict[str, float]] = {}
    for alias in model_aliases:
        own_scores = per_model_question_means.get(alias, [])
        peers = [a for a in model_aliases if a != alias]

        consistency_vals = []
        for peer in peers:
            consistency_vals.append(
                _pearson(own_scores, per_model_question_means.get(peer, [])))
        consistency = 50.0 if not consistency_vals else _bounded(
            (mean(consistency_vals) + 1.0) * 50.0)

        variance_penalty = _variance(own_scores)
        variance_score = _bounded(100.0 - variance_penalty)

        severity_bias = abs(all_avg_scores.get(alias, 0.0) - global_mean)
        severity_bias_score = _bounded(100.0 - (severity_bias * 2.0))

        skip_scores = per_model_skip_scores.get(alias, [])
        skip_handling_accuracy = 50.0 if not skip_scores else _bounded(
            mean(skip_scores))

        spread_dims = []
        output = score_outputs_by_model[alias]
        for row in output.get("question_scores", []):
            dims = [float(row.get("relevance", 0)), float(
                row.get("depth", 0)), float(row.get("clarity", 0))]
            spread_dims.append(max(dims) - min(dims))
        spread_score = _bounded(
            100.0 - (mean(spread_dims) if spread_dims else 0.0))

        sms = _bounded(
            0.25 * consistency
            + 0.20 * variance_score
            + 0.20 * severity_bias_score
            + 0.20 * skip_handling_accuracy
            + 0.15 * spread_score
        )

        result[alias] = {
            "score_consistency": round(consistency, 4),
            "score_variance": round(variance_score, 4),
            "severity_bias": round(severity_bias_score, 4),
            "skip_handling_accuracy": round(skip_handling_accuracy, 4),
            "multi_dimensional_spread": round(spread_score, 4),
            "sms": round(sms, 4),
        }

    return result
