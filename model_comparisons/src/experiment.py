from __future__ import annotations
from pipeline.scorer import score_interview
from pipeline.resume_processor import (
    chunk_text,
    extract_resume_insights_with_llm,
    extract_text_from_pdf,
    generate_embeddings,
    initialize_chroma_db,
    initialize_datasets_in_chromadb,
    retrieve_resume_context,
    store_jd_in_chroma,
    store_resume_in_chroma,
)
from pipeline.question_generator import generate_questions
from pipeline.jd_parser import match_to_company_profile, parse_jd_with_llm

import argparse
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config_loader import load_combinations, load_experiment_config, load_judge_rubrics
from .hash_utils import sha256_of_payload
from .io_utils import read_json, read_text, write_csv, write_json
from .llm_client import ModelConfig, OpenAICompatibleClient, ProviderConfig
from .metrics import question_metrics, scoring_metrics
from .paths import RunPaths, create_run_paths
from .prompting import load_prompt, render_prompt
from .reporting import write_report
from .stats import correlation_matrix, one_way_anova, pairwise_t_tests
from .validation import validate_270_answers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _read_resume_context(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(path))
    return read_text(path)


def _load_contexts(base_dir: Path, combo: dict[str, Any]) -> tuple[str, str]:
    resume_path = base_dir / "data" / combo["resume_file"]
    jd_path = base_dir / "data" / combo["jd_file"]
    return _read_resume_context(resume_path), read_text(jd_path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_from_dict(item: dict[str, Any]) -> ModelConfig:
    return ModelConfig(model_id=item["id"], alias=item["alias"], provider=item["provider"])


def _provider_map(config: dict[str, Any]) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for provider_name, values in config["providers"].items():
        providers[provider_name] = ProviderConfig(
            name=provider_name,
            base_url=values["base_url"],
            api_key_env=values["api_key_env"],
        )
    return providers


def _question_filename(combo_id: str, model_alias: str) -> str:
    return f"{combo_id}_{model_alias}.json"


def _score_filename(combo_id: str, source_alias: str, evaluator_alias: str) -> str:
    return f"{combo_id}_{source_alias}_by_{evaluator_alias}_scores.json"


def _judge_scoring_filename(combo_id: str, source_alias: str) -> str:
    return f"{combo_id}_{source_alias}_judge.json"


class PipelineLLMAdapter:
    """Adapter that exposes .invoke(prompt) for the existing pipeline functions."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: ModelConfig,
        *,
        temperature: float,
        max_tokens: int = 4000,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt_text: str) -> str:
        return self.client.completion_text(
            self.model,
            prompt_text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt="Return valid JSON only.",
        )


def _to_question_dicts(combo_id: str, model: ModelConfig, questions: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        question_text = str(getattr(item, "question", "")).strip()
        q_type = str(getattr(item, "type", "technical")).strip().lower()
        difficulty = str(getattr(item, "difficulty", "medium")).strip().lower()
        focus_area = str(getattr(item, "focus_area", "mixed")).strip().lower()
        rationale = str(getattr(item, "rationale", "")).strip()
        hints = getattr(item, "hint", [])
        if not isinstance(hints, list):
            hints = [str(hints)]

        rows.append(
            {
                "question_id": f"{combo_id}_{model.alias}_q{idx:02d}",
                "question_text": question_text,
                "question": question_text,
                "type": q_type,
                "difficulty": difficulty,
                "focus_area": focus_area,
                "hint": [str(h).strip() for h in hints if str(h).strip()],
                "hints": [str(h).strip() for h in hints if str(h).strip()],
                "rationale": rationale,
            }
        )
    return rows


def _to_qa_pairs_from_answers(answer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    qa_pairs: list[dict[str, Any]] = []
    for item in answer_payload.get("responses", []):
        label = str(item.get("answer_label", "")).upper()
        answer_text = str(item.get("answer_text", ""))
        is_skipped = label == "SKIPPED" or answer_text.strip() == ""
        qa_pairs.append(
            {
                "question": str(item.get("question_text", "")),
                "answer": "[SKIPPED]" if is_skipped else answer_text,
                "type": str(item.get("question_type", "technical")) or "technical",
                "difficulty": str(item.get("question_difficulty", "medium")) or "medium",
                "was_skipped": is_skipped,
            }
        )
    return qa_pairs


def _normalize_score_output(interview_scores: Any, answer_payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    responses = answer_payload.get("responses", [])
    per_answer_scores = list(
        getattr(interview_scores, "per_answer_scores", []))

    for idx, score_item in enumerate(per_answer_scores):
        response_row = responses[idx] if idx < len(responses) else {}
        rows.append(
            {
                "question_id": str(response_row.get("question_id", f"q_{idx+1:02d}")),
                "question_text": str(response_row.get("question_text", getattr(score_item, "question", ""))),
                "relevance": float(getattr(score_item, "relevance_score", 0.0)),
                "depth": float(getattr(score_item, "depth_score", 0.0)),
                "clarity": float(getattr(score_item, "clarity_score", 0.0)),
                "feedback": str(getattr(score_item, "feedback", "")),
                "was_skipped": bool(getattr(score_item, "was_skipped", False)),
            }
        )

    return {
        "question_scores": rows,
        "aggregate": {
            "overall": float(getattr(interview_scores, "overall_score", 0.0)),
            "technical_depth": float(getattr(interview_scores, "technical_depth_score", 0.0)),
            "communication": float(getattr(interview_scores, "communication_score", 0.0)),
            "readiness": float(getattr(interview_scores, "role_readiness_percent", 0.0)),
            "readiness_label": str(getattr(interview_scores, "role_readiness_label", "")),
        },
        "what_went_well": list(getattr(interview_scores, "what_went_well", [])),
        "areas_to_improve": list(getattr(interview_scores, "areas_to_improve", [])),
        "recommended_resources": list(getattr(interview_scores, "recommended_resources", [])),
    }


class ModelComparisonExperiment:
    def __init__(self, base_dir: Path, run_id: str | None) -> None:
        self.base_dir = base_dir
        self.config = load_experiment_config(base_dir)
        self.rubrics = load_judge_rubrics(base_dir)
        self.combinations = load_combinations(base_dir)
        self.paths = create_run_paths(base_dir, run_id)
        self.repo_root = base_dir.parent.resolve()

        self.eval_models = [_model_from_dict(
            m) for m in self.config["models_under_evaluation"]]
        self.judge_model = _model_from_dict(self.config["judge_model"])

        retry_cfg = self.config["experiment"]["retry"]
        self.client = OpenAICompatibleClient(
            providers=_provider_map(self.config),
            request_delay_seconds=float(
                self.config["experiment"]["request_delay_seconds"]),
            max_retries=int(retry_cfg["max_retries"]),
            backoff_base_seconds=float(retry_cfg["backoff_base_seconds"]),
            temperature=0.2,
            max_tokens=4000,
        )

        self.prompt_hashes_path = self.paths.manifests / "prompt_hashes.json"
        self.metadata_path = self.paths.manifests / "metadata.json"
        self.judge_mappings_path = self.paths.manifests / "judge_mappings.json"

        if not self.prompt_hashes_path.exists():
            write_json(self.prompt_hashes_path, {
                       "generation": {}, "scoring": {}})
        if not self.metadata_path.exists():
            write_json(
                self.metadata_path,
                {
                    "run_id": self.paths.run_id,
                    "difficulty": self.config["experiment"]["difficulty"],
                    "focus_area": self.config["experiment"]["focus_area"],
                    "num_questions": self.config["experiment"]["num_questions"],
                    "evaluated_models": [m.model_id for m in self.eval_models],
                    "judge_model": self.judge_model.model_id,
                },
            )
        if not self.judge_mappings_path.exists():
            write_json(self.judge_mappings_path, {
                       "question_phase": {}, "scoring_phase": {}})

    def run_prepare(self) -> None:
        with _temporary_cwd(self.repo_root):
            client = initialize_chroma_db()
            initialize_datasets_in_chromadb(client)
        write_json(self.paths.manifests /
                   "combinations_snapshot.json", self.combinations)

    def run_generation(self) -> None:
        hashes = read_json(self.prompt_hashes_path)

        with _temporary_cwd(self.repo_root):
            client = initialize_chroma_db()
            initialize_datasets_in_chromadb(client)

            for combo in self.combinations:
                combo_id = combo["combo_id"]
                resume_text, jd_text = _load_contexts(self.base_dir, combo)
                chunks = chunk_text(resume_text)
                embeddings, _ = generate_embeddings(chunks)

                canonical_payload = {
                    "combo_id": combo_id,
                    "resume_hash": _sha256_text(resume_text),
                    "jd_hash": _sha256_text(jd_text),
                    "difficulty": self.config["experiment"]["difficulty"],
                    "focus_area": self.config["experiment"]["focus_area"],
                    "num_questions": self.config["experiment"]["num_questions"],
                }
                input_hash = sha256_of_payload(canonical_payload)

                for model in self.eval_models:
                    session_id = f"{self.paths.run_id}_{combo_id}_{model.alias}".lower().replace(
                        "/", "_")
                    store_resume_in_chroma(
                        chunks, embeddings, session_id, client)
                    store_jd_in_chroma(jd_text, session_id, client)

                    parse_llm = PipelineLLMAdapter(
                        self.client, model, temperature=0.5)
                    question_llm = PipelineLLMAdapter(
                        self.client, model, temperature=0.7)
                    resume_llm = PipelineLLMAdapter(
                        self.client, model, temperature=0.3)

                    resume_insights = extract_resume_insights_with_llm(
                        resume_text,
                        llm=resume_llm,
                        prompt_path=str(
                            self.repo_root / "prompts" / "resume_insights_prompt.txt"),
                    )

                    parsed_jd = parse_jd_with_llm(
                        jd_text,
                        llm=parse_llm,
                        chroma_client=client,
                        k_examples=3,
                    )

                    company_profile = match_to_company_profile(
                        parsed_jd, client)
                    retrieval_query = " ".join(parsed_jd.get(
                        "skills", [])) or parsed_jd.get("role", "")
                    resume_context = retrieve_resume_context(
                        query=retrieval_query,
                        session_id=session_id,
                        k=5,
                        client=client,
                    )

                    questions = generate_questions(
                        parsed_jd=parsed_jd,
                        company_profile=company_profile,
                        resume_context=resume_context,
                        preferences={
                            "difficulty": self.config["experiment"]["difficulty"],
                            "num_questions": int(self.config["experiment"]["num_questions"]),
                            "focus_area": self.config["experiment"]["focus_area"],
                        },
                        chroma_client=client,
                        resume_text=resume_text,
                        llm=question_llm,
                    )

                    serialized_questions = _to_question_dicts(
                        combo_id, model, questions)
                    output_payload = {
                        "combo_id": combo_id,
                        "model_alias": model.alias,
                        "model_id": model.model_id,
                        "pipeline_flow": "normal_session_parity",
                        "input_hash": input_hash,
                        "resume_context": resume_context,
                        "retrieval_query": retrieval_query,
                        "parsed_jd": parsed_jd,
                        "company_profile": company_profile,
                        "resume_insights": resume_insights,
                        "questions": serialized_questions,
                    }
                    write_json(
                        self.paths.generated_questions /
                        _question_filename(combo_id, model.alias),
                        output_payload,
                    )
                    hashes["generation"][f"{combo_id}:{model.alias}"] = input_hash

        write_json(self.prompt_hashes_path, hashes)

    def scaffold_human_answer_files(self) -> None:
        for combo in self.combinations:
            combo_id = combo["combo_id"]
            for model in self.eval_models:
                source_file = self.paths.generated_questions / \
                    _question_filename(combo_id, model.alias)
                if not source_file.exists():
                    continue

                generated = read_json(source_file)
                responses = []
                for q in generated.get("questions", []):
                    responses.append(
                        {
                            "question_id": q.get("question_id", ""),
                            "question_text": q.get("question_text", q.get("question", "")),
                            "question_type": q.get("type", "technical"),
                            "question_difficulty": q.get("difficulty", self.config["experiment"]["difficulty"]),
                            "answer_text": "",
                            "answer_label": "SKIPPED",
                            "timestamp_utc": "",
                        }
                    )

                answer_payload = {
                    "combo_id": combo_id,
                    "source_model_alias": model.alias,
                    "source_model_id": model.model_id,
                    "difficulty": self.config["experiment"]["difficulty"],
                    "focus_area": self.config["experiment"]["focus_area"],
                    "num_questions": self.config["experiment"]["num_questions"],
                    "responses": responses,
                }
                path = self.paths.human_answers / \
                    f"{combo_id}_{model.alias}_answers.json"
                if not path.exists():
                    write_json(path, answer_payload)

    def run_scoring(self) -> None:
        hashes = read_json(self.prompt_hashes_path)

        with _temporary_cwd(self.repo_root):
            for combo in self.combinations:
                combo_id = combo["combo_id"]
                for source_model in self.eval_models:
                    answers_file = self.paths.human_answers / \
                        f"{combo_id}_{source_model.alias}_answers.json"
                    if not answers_file.exists():
                        raise RuntimeError(
                            f"Missing answer file: {answers_file}")

                    source_generated = read_json(
                        self.paths.generated_questions /
                        _question_filename(combo_id, source_model.alias)
                    )
                    answer_payload = read_json(answers_file)
                    scoring_hash = sha256_of_payload(answer_payload)
                    hashes["scoring"][f"{combo_id}:{source_model.alias}"] = scoring_hash

                    role = str(source_generated.get(
                        "parsed_jd", {}).get("role", "Unknown Role"))
                    company = str(source_generated.get(
                        "company_profile", {}).get("company", "Unknown Company"))
                    qa_pairs = _to_qa_pairs_from_answers(answer_payload)

                    for evaluator_model in self.eval_models:
                        scorer_llm = PipelineLLMAdapter(
                            self.client, evaluator_model, temperature=0.3)
                        interview_scores = score_interview(
                            qa_pairs=qa_pairs,
                            role=role,
                            company=company,
                            resume_insights=source_generated.get(
                                "resume_insights", {}),
                            jd_insights=source_generated.get("parsed_jd", {}),
                            generated_questions=source_generated.get(
                                "questions", []),
                            llm=scorer_llm,
                            prompt_path=str(self.repo_root /
                                            "prompts" / "scorer_prompt.txt"),
                        )

                        normalized_scoring = _normalize_score_output(
                            interview_scores, answer_payload)
                        output_payload = {
                            "combo_id": combo_id,
                            "source_model_alias": source_model.alias,
                            "evaluator_model_alias": evaluator_model.alias,
                            "evaluator_model_id": evaluator_model.model_id,
                            "pipeline_flow": "normal_session_parity",
                            "prompt_hash": scoring_hash,
                            "scoring": normalized_scoring,
                        }
                        write_json(
                            self.paths.model_scores
                            / _score_filename(combo_id, source_model.alias, evaluator_model.alias),
                            output_payload,
                        )

        write_json(self.prompt_hashes_path, hashes)

    def run_judge_question_phase(self) -> None:
        template = load_prompt(self.base_dir, "judge_question_prompt.txt")
        mappings = read_json(self.judge_mappings_path)
        rnd = random.Random(int(self.config["experiment"]["random_seed"]))

        for combo in self.combinations:
            combo_id = combo["combo_id"]
            resume_context, jd_context = _load_contexts(self.base_dir, combo)
            outputs_by_alias = {}
            for model in self.eval_models:
                path = self.paths.generated_questions / \
                    _question_filename(combo_id, model.alias)
                outputs_by_alias[model.alias] = read_json(
                    path).get("questions", [])

            aliases = [m.alias for m in self.eval_models]
            rnd.shuffle(aliases)
            mapping = {"A": aliases[0], "B": aliases[1], "C": aliases[2]}

            prompt = render_prompt(
                template,
                {
                    "resume_summary": resume_context[:6000],
                    "jd_summary": jd_context[:4000],
                    "system_a_output": json.dumps(outputs_by_alias[mapping["A"]], ensure_ascii=True),
                    "system_b_output": json.dumps(outputs_by_alias[mapping["B"]], ensure_ascii=True),
                    "system_c_output": json.dumps(outputs_by_alias[mapping["C"]], ensure_ascii=True),
                },
            )
            judge_raw = self.client.completion_json(self.judge_model, prompt)

            systems = judge_raw.get("systems", {})
            unblinded_scores = {
                mapping[key]: float(systems.get(key, {}).get("score", 0.0)) for key in ["A", "B", "C"]
            }
            payload = {
                "combo_id": combo_id,
                "judge_model": self.judge_model.model_id,
                "mapping": mapping,
                "raw": judge_raw,
                "model_scores": unblinded_scores,
                "confidence": float(judge_raw.get("confidence", 0.0)),
            }
            write_json(self.paths.judge_question_phase /
                       f"{combo_id}_judge.json", payload)
            mappings["question_phase"][combo_id] = mapping

        write_json(self.judge_mappings_path, mappings)

    def run_judge_scoring_phase(self) -> None:
        template = load_prompt(self.base_dir, "judge_scoring_prompt.txt")
        mappings = read_json(self.judge_mappings_path)
        rnd = random.Random(
            int(self.config["experiment"]["random_seed"]) + 1000)

        for combo in self.combinations:
            combo_id = combo["combo_id"]
            for source_model in self.eval_models:
                answers_file = self.paths.human_answers / \
                    f"{combo_id}_{source_model.alias}_answers.json"
                answer_payload = read_json(answers_file)

                outputs_by_alias = {}
                for evaluator_model in self.eval_models:
                    path = self.paths.model_scores / _score_filename(
                        combo_id, source_model.alias, evaluator_model.alias
                    )
                    outputs_by_alias[evaluator_model.alias] = read_json(
                        path).get("scoring", {})

                aliases = [m.alias for m in self.eval_models]
                rnd.shuffle(aliases)
                mapping = {"A": aliases[0], "B": aliases[1], "C": aliases[2]}

                prompt = render_prompt(
                    template,
                    {
                        "qa_payload": json.dumps(answer_payload, ensure_ascii=True),
                        "system_a_output": json.dumps(outputs_by_alias[mapping["A"]], ensure_ascii=True),
                        "system_b_output": json.dumps(outputs_by_alias[mapping["B"]], ensure_ascii=True),
                        "system_c_output": json.dumps(outputs_by_alias[mapping["C"]], ensure_ascii=True),
                    },
                )
                judge_raw = self.client.completion_json(
                    self.judge_model, prompt)

                systems = judge_raw.get("systems", {})
                unblinded_scores = {
                    mapping[key]: float(systems.get(key, {}).get("score", 0.0)) for key in ["A", "B", "C"]
                }
                payload = {
                    "combo_id": combo_id,
                    "source_model_alias": source_model.alias,
                    "judge_model": self.judge_model.model_id,
                    "mapping": mapping,
                    "raw": judge_raw,
                    "model_scores": unblinded_scores,
                    "confidence": float(judge_raw.get("confidence", 0.0)),
                }
                write_json(
                    self.paths.judge_scoring_phase /
                    _judge_scoring_filename(combo_id, source_model.alias),
                    payload,
                )
                mappings["scoring_phase"][f"{combo_id}:{source_model.alias}"] = mapping

        write_json(self.judge_mappings_path, mappings)

    def run_analysis_and_report(self) -> None:
        q_weights = self.config["scoring_weights"]["question_phase"]
        s_weights = self.config["scoring_weights"]["scoring_phase"]
        f_weights = self.config["scoring_weights"]["final"]

        question_phase_rows: list[dict[str, Any]] = []
        scoring_phase_rows: list[dict[str, Any]] = []
        judge_rows: list[dict[str, Any]] = []

        per_model_qps: dict[str, list[float]] = defaultdict(list)
        per_model_sps: dict[str, list[float]] = defaultdict(list)

        for combo in self.combinations:
            combo_id = combo["combo_id"]
            resume_context, jd_context = _load_contexts(self.base_dir, combo)
            judge_q_file = self.paths.judge_question_phase / \
                f"{combo_id}_judge.json"
            judge_q_payload = read_json(judge_q_file)
            judge_q_scores = judge_q_payload.get("model_scores", {})

            for model in self.eval_models:
                generated = read_json(
                    self.paths.generated_questions /
                    _question_filename(combo_id, model.alias)
                )
                questions = generated.get("questions", [])
                metrics = question_metrics(
                    questions, resume_context, jd_context)

                qms = float(metrics["qms"])
                jqs = float(judge_q_scores.get(model.alias, 0.0))
                qps = float(q_weights["qms_weight"]) * \
                    qms + float(q_weights["jqs_weight"]) * jqs

                row = {
                    "combo_id": combo_id,
                    "model_alias": model.alias,
                    "qms": qms,
                    "jqs": jqs,
                    "qps": qps,
                    **metrics,
                }
                question_phase_rows.append(row)
                per_model_qps[model.alias].append(qps)

                judge_rows.append(
                    {
                        "phase": "question",
                        "combo_id": combo_id,
                        "source_model_alias": "-",
                        "model_alias": model.alias,
                        "judge_score": jqs,
                        "judge_confidence": float(judge_q_payload.get("confidence", 0.0)),
                    }
                )

        for combo in self.combinations:
            combo_id = combo["combo_id"]
            for source_model in self.eval_models:
                answers_file = self.paths.human_answers / \
                    f"{combo_id}_{source_model.alias}_answers.json"
                answer_payload = read_json(answers_file)

                outputs_by_model = {}
                for evaluator_model in self.eval_models:
                    raw = read_json(
                        self.paths.model_scores
                        / _score_filename(combo_id, source_model.alias, evaluator_model.alias)
                    )
                    outputs_by_model[evaluator_model.alias] = raw.get(
                        "scoring", {})

                metric_bundle = scoring_metrics(
                    outputs_by_model, answer_payload)
                judge_s_file = self.paths.judge_scoring_phase / \
                    _judge_scoring_filename(combo_id, source_model.alias)
                judge_s_payload = read_json(judge_s_file)
                judge_s_scores = judge_s_payload.get("model_scores", {})

                for evaluator_model in self.eval_models:
                    sms = float(metric_bundle[evaluator_model.alias]["sms"])
                    jss = float(judge_s_scores.get(evaluator_model.alias, 0.0))
                    sps = float(s_weights["sms_weight"]) * \
                        sms + float(s_weights["jss_weight"]) * jss
                    row = {
                        "combo_id": combo_id,
                        "source_model_alias": source_model.alias,
                        "model_alias": evaluator_model.alias,
                        "sms": sms,
                        "jss": jss,
                        "sps": sps,
                        **metric_bundle[evaluator_model.alias],
                    }
                    scoring_phase_rows.append(row)
                    per_model_sps[evaluator_model.alias].append(sps)

                    judge_rows.append(
                        {
                            "phase": "scoring",
                            "combo_id": combo_id,
                            "source_model_alias": source_model.alias,
                            "model_alias": evaluator_model.alias,
                            "judge_score": jss,
                            "judge_confidence": float(judge_s_payload.get("confidence", 0.0)),
                        }
                    )

        ranking = []
        for model in self.eval_models:
            qps_avg = mean(per_model_qps[model.alias]
                           ) if per_model_qps[model.alias] else 0.0
            sps_avg = mean(per_model_sps[model.alias]
                           ) if per_model_sps[model.alias] else 0.0
            final_score = float(f_weights["qps_average_weight"]) * qps_avg + float(
                f_weights["sps_average_weight"]
            ) * sps_avg
            ranking.append(
                {
                    "model_alias": model.alias,
                    "avg_question_phase_score": qps_avg,
                    "avg_scoring_phase_score": sps_avg,
                    "final_score": final_score,
                    "stability_std_proxy": 0.0,
                }
            )

        ranking.sort(key=lambda x: x["final_score"], reverse=True)

        score_groups = {
            m.alias: per_model_sps[m.alias] for m in self.eval_models}
        anova_result = one_way_anova(score_groups)
        pairwise = pairwise_t_tests(score_groups)
        corr = correlation_matrix(score_groups)

        write_csv(
            self.paths.analysis / "metric_scores.csv",
            question_phase_rows + scoring_phase_rows,
            fieldnames=sorted(
                {k for row in (question_phase_rows + scoring_phase_rows) for k in row.keys()}),
        )
        write_csv(
            self.paths.analysis / "judge_scores.csv",
            judge_rows,
            fieldnames=["phase", "combo_id", "source_model_alias",
                        "model_alias", "judge_score", "judge_confidence"],
        )

        fused_rows = []
        for row in ranking:
            fused_rows.append(
                {
                    "model_alias": row["model_alias"],
                    "avg_question_phase_score": row["avg_question_phase_score"],
                    "avg_scoring_phase_score": row["avg_scoring_phase_score"],
                    "final_score": row["final_score"],
                }
            )
        write_csv(
            self.paths.analysis / "fused_scores.csv",
            fused_rows,
            fieldnames=["model_alias", "avg_question_phase_score",
                        "avg_scoring_phase_score", "final_score"],
        )

        stats_summary = {
            "anova": anova_result,
            "pairwise_tests": pairwise,
            "correlation_matrix": corr,
            "question_phase_count": len(question_phase_rows),
            "scoring_phase_count": len(scoring_phase_rows),
        }
        write_json(self.paths.analysis / "stats_summary.json", stats_summary)

        report_summary = {
            "question_phase_rows": question_phase_rows,
            "scoring_phase_rows": scoring_phase_rows,
            "final_ranking": ranking,
            "anova": anova_result,
            "pairwise_tests": pairwise,
        }
        write_report(self.paths.reports / "report.md", report_summary)

    def run_validate_answers(self) -> None:
        summary = validate_270_answers(self.paths.human_answers)
        write_json(self.paths.analysis /
                   "answer_collection_validation.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True))

    def run_all(self) -> None:
        self.run_prepare()
        self.run_generation()
        self.scaffold_human_answer_files()
        self.run_scoring()
        self.run_judge_question_phase()
        self.run_judge_scoring_phase()
        self.run_analysis_and_report()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InterviewAI model comparison runner")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to model_comparisons directory",
    )
    parser.add_argument("--run-id", default=None,
                        help="Optional run id, e.g. run_20260415_120000")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "prepare",
            "generate",
            "scaffold_answers",
            "validate_answers",
            "score",
            "judge_questions",
            "judge_scoring",
            "analyze",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()
    runner = ModelComparisonExperiment(base_dir, args.run_id)

    if args.stage == "prepare":
        runner.run_prepare()
    elif args.stage == "generate":
        runner.run_generation()
    elif args.stage == "scaffold_answers":
        runner.scaffold_human_answer_files()
    elif args.stage == "validate_answers":
        runner.run_validate_answers()
    elif args.stage == "score":
        runner.run_scoring()
    elif args.stage == "judge_questions":
        runner.run_judge_question_phase()
    elif args.stage == "judge_scoring":
        runner.run_judge_scoring_phase()
    elif args.stage == "analyze":
        runner.run_analysis_and_report()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
