from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   use_api: bool = False) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    empty_result = {
        **{name: 0.0 for name in metric_names},
        "per_question": [],
        "num_questions": len(questions),
        "evaluation_status": "skipped",
    }
    if not use_api:
        print("  ⚠️  RAGAS skipped: API evaluation was not explicitly enabled.")
        return empty_result
    if not LLM_API_KEY:
        print("  ⚠️  RAGAS skipped: LLM_API_KEY is not configured.")
        return empty_result
    if not questions or not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        print("  ⚠️  RAGAS skipped: evaluation inputs are empty or have different lengths.")
        return empty_result

    try:
        import math
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        llm_options = {"model": LLM_MODEL, "api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            llm_options["base_url"] = LLM_BASE_URL
        llm = ChatOpenAI(**llm_options)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        dataframe = result.to_pandas()

        def score_or_zero(value):
            try:
                number = float(value)
                return number if math.isfinite(number) else 0.0
            except (TypeError, ValueError):
                return 0.0

        per_question = []
        for index, row in dataframe.iterrows():
            per_question.append(EvalResult(
                question=questions[index], answer=answers[index], contexts=contexts[index],
                ground_truth=ground_truths[index],
                faithfulness=score_or_zero(row.get("faithfulness", 0.0)),
                answer_relevancy=score_or_zero(row.get("answer_relevancy", 0.0)),
                context_precision=score_or_zero(row.get("context_precision", 0.0)),
                context_recall=score_or_zero(row.get("context_recall", 0.0)),
            ))
        aggregate = {
            name: score_or_zero(dataframe[name].mean()) if name in dataframe else 0.0
            for name in metric_names
        }
        return {
            **aggregate,
            "per_question": per_question,
            "num_questions": len(questions),
            "evaluation_status": "completed",
        }
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return {**empty_result, "evaluation_status": "failed"}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("Câu trả lời có thông tin không được context hỗ trợ.",
                         "Siết prompt chỉ trả lời theo context và giảm temperature."),
        "context_recall": ("Context thiếu thông tin cần để trả lời.",
                           "Cải thiện chunking, truy hồi hybrid hoặc tăng candidate top-k."),
        "context_precision": ("Các context truy hồi chứa nhiều đoạn không liên quan.",
                              "Bổ sung reranking hoặc lọc metadata trước khi tạo câu trả lời."),
        "answer_relevancy": ("Câu trả lời chưa giải quyết đúng trọng tâm câu hỏi.",
                             "Cải thiện prompt trả lời và kiểm tra query/context được truyền vào."),
    }
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name, 0.0) or 0.0) for name in metric_names}
        worst_metric = min(metric_names, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        average_score = sum(scores.values()) / len(metric_names)
        failures.append({
            "question": result.question,
            "expected": result.ground_truth,
            "got": result.answer,
            "contexts": result.contexts,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "average_score": average_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": (
                f"Output sai → kiểm tra context → kiểm tra truy hồi → {diagnosis} "
                f"Fix: {suggested_fix}"
            ),
        })
    failures.sort(key=lambda failure: failure["average_score"])
    return failures[:max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {
            k: v for k, v in results.items()
            if k not in {"per_question", "num_questions", "evaluation_status"}
        },
        "num_questions": results.get("num_questions", len(results.get("per_question", []))),
        "evaluation_status": results.get("evaluation_status", "completed"),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
