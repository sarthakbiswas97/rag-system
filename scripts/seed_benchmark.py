#!/usr/bin/env python3
"""Seed benchmark data for load testing.

Creates a test tenant and ingests synthetic documents so the
load test has data to query against.

Usage:
    uv run python scripts/seed_benchmark.py --num-docs 100
    uv run python scripts/seed_benchmark.py --num-docs 1000 --host http://localhost:8000
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import httpx

TOPICS = [
    "machine learning",
    "neural networks",
    "gradient descent",
    "backpropagation",
    "transformer architecture",
    "attention mechanisms",
    "convolutional networks",
    "recurrent networks",
    "reinforcement learning",
    "natural language processing",
    "computer vision",
    "generative models",
    "transfer learning",
    "fine-tuning",
    "embeddings",
    "tokenization",
    "loss functions",
    "optimization",
    "regularization",
    "batch normalization",
]

TEMPLATES = [
    "{topic} is a fundamental concept in artificial intelligence. "
    "It involves {detail}. Researchers have developed numerous "
    "approaches to improve {topic} performance over the years.",
    "The field of {topic} has seen rapid advancement in recent years. "
    "Key techniques include {detail}. These methods have been "
    "applied across various domains with significant success.",
    "Understanding {topic} requires knowledge of {detail}. "
    "In practice, {topic} is used in applications ranging from "
    "healthcare to autonomous vehicles.",
    "Recent breakthroughs in {topic} have been driven by {detail}. "
    "The scalability of modern {topic} approaches has enabled "
    "training on datasets of unprecedented size.",
]

DETAILS = [
    "mathematical optimization and statistical inference",
    "large-scale parallel computation on GPU clusters",
    "careful hyperparameter tuning and architecture search",
    "novel training strategies and data augmentation techniques",
    "efficient memory management and gradient checkpointing",
    "distributed training across multiple nodes",
    "self-supervised and contrastive learning objectives",
    "attention-based mechanisms and positional encodings",
]


def generate_document(doc_id: int) -> tuple[str, str]:
    """Generate a synthetic document with consistent content."""
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)
    detail = random.choice(DETAILS)

    paragraphs = []
    for _ in range(random.randint(3, 8)):
        text = template.format(topic=topic, detail=detail)
        paragraphs.append(text)

    content = "\n\n".join(paragraphs)
    filename = f"benchmark_doc_{doc_id:06d}_{topic.replace(' ', '_')}.txt"
    return filename, content


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed benchmark data")
    parser.add_argument(
        "--num-docs", type=int, default=100, help="Number of documents to create"
    )
    parser.add_argument(
        "--host", type=str, default="http://localhost:8000", help="API host"
    )
    parser.add_argument(
        "--admin-key", type=str, default="", help="Admin API key"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Documents per ingest request"
    )
    args = parser.parse_args()

    client = httpx.Client(base_url=args.host, timeout=120.0)

    # Step 1: Register a benchmark tenant
    print("Registering benchmark tenant...")
    resp = client.post(
        "/v1/register",
        json={"name": f"benchmark-{int(time.time())}", "email": "bench@test.com"},
    )
    if resp.status_code != 201:
        print(f"Failed to register tenant: {resp.status_code} {resp.text}")
        sys.exit(1)

    data = resp.json()
    api_key = data["api_key"]
    tenant_id = data["tenant"]["id"]
    print(f"  Tenant: {tenant_id}")
    print(f"  API Key: {api_key[:20]}...")

    # Step 2: Generate and ingest documents in batches
    print(f"\nIngesting {args.num_docs} documents in batches of {args.batch_size}...")
    total_chunks = 0
    total_time = 0.0
    docs_ingested = 0

    for batch_start in range(0, args.num_docs, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.num_docs)
        files = []
        for doc_id in range(batch_start, batch_end):
            filename, content = generate_document(doc_id)
            files.append(("files", (filename, content.encode(), "text/plain")))

        start = time.perf_counter()
        resp = client.post(
            "/v1/ingest",
            files=files,
            headers={"X-API-Key": api_key},
        )
        elapsed = time.perf_counter() - start

        if resp.status_code == 200:
            result = resp.json()
            total_chunks += result["chunks_created"]
            total_time += elapsed
            docs_ingested += result["documents_processed"]
            print(
                f"  Batch {batch_start}-{batch_end}: "
                f"{result['chunks_created']} chunks in {elapsed:.1f}s"
            )
        else:
            print(f"  Batch {batch_start}-{batch_end}: FAILED {resp.status_code}")

    print("\nSeed complete:")
    print(f"  Documents ingested: {docs_ingested}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Throughput: {docs_ingested / total_time:.1f} docs/s")
    print("\nAPI key for load testing:")
    print(f"  export RAG_API_KEY={api_key}")


if __name__ == "__main__":
    main()
