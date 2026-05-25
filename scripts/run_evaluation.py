"""Offline RAG evaluation script.

Usage:
    uv run python scripts/run_evaluation.py --dataset eval_data.json

Dataset format (JSON):
[
    {
        "question": "What is X?",
        "answer": "X is Y [1].",
        "contexts": ["Source doc about X..."],
        "ground_truth": "X is Y."  // optional
    }
]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.config import get_settings
from rag.evaluation.metrics import EvalSample, RAGEvaluator
from rag.verification.entailment import EntailmentChecker

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[EvalSample]:
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(
            EvalSample(
                question=item["question"],
                answer=item["answer"],
                contexts=tuple(item.get("contexts", [])),
                ground_truth=item.get("ground_truth"),
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to evaluation dataset (JSON)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum faithfulness threshold to pass (default: 0.95)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()

    logger.info("Loading NLI model: %s", settings.nli_model)
    checker = EntailmentChecker(model_name=settings.nli_model)

    logger.info("Loading dataset: %s", args.dataset)
    samples = load_dataset(args.dataset)
    logger.info("Loaded %d samples", len(samples))

    evaluator = RAGEvaluator(entailment_checker=checker)
    report = evaluator.evaluate(samples)

    print("\n=== RAG Evaluation Report ===")
    print(f"Samples:               {report.total_samples}")
    print(f"Mean Faithfulness:     {report.mean_faithfulness:.3f}")
    print(f"Mean Context Util:     {report.mean_context_utilization:.3f}")
    if report.mean_citation_precision is not None:
        print(f"Mean Citation Prec:    {report.mean_citation_precision:.3f}")
    print(f"Threshold:             {args.threshold:.3f}")

    if report.mean_faithfulness >= args.threshold:
        print("\nRESULT: PASS")
    else:
        print(
            f"\nRESULT: FAIL (faithfulness {report.mean_faithfulness:.3f} "
            f"< {args.threshold:.3f})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
