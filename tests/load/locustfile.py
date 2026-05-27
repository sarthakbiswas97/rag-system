"""Load testing suite for RAG API.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Or headless mode:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 --run-time 60s

Environment variables:
    RAG_API_KEY: tenant API key for authenticated endpoints
    RAG_LOAD_TEST_QUESTIONS: path to a JSON file with test questions
"""

from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, task

API_KEY = os.getenv("RAG_API_KEY", "rk_test_key")

DEFAULT_QUESTIONS = [
    "What is machine learning?",
    "How does gradient descent work?",
    "Explain transformer architecture",
    "What is backpropagation?",
    "How do attention mechanisms work?",
    "What is a neural network?",
    "Explain the difference between supervised and unsupervised learning",
    "What is reinforcement learning?",
    "How does batch normalization work?",
    "What is dropout regularization?",
    "Explain the concept of overfitting",
    "What is transfer learning?",
    "How do convolutional neural networks work?",
    "What is a recurrent neural network?",
    "Explain the concept of embeddings",
    "What is fine-tuning in deep learning?",
    "How does BERT work?",
    "What is the difference between precision and recall?",
    "Explain cross-entropy loss",
    "What is a learning rate scheduler?",
]


def _load_questions() -> list[str]:
    questions_path = os.getenv("RAG_LOAD_TEST_QUESTIONS")
    if questions_path and os.path.exists(questions_path):
        with open(questions_path) as f:
            return json.load(f)
    return DEFAULT_QUESTIONS


QUESTIONS = _load_questions()


class RAGQueryUser(HttpUser):
    """Simulates users making query requests."""

    wait_time = between(0.5, 2.0)
    weight = 8  # 80% of traffic is queries

    @task(10)
    def query(self) -> None:
        question = random.choice(QUESTIONS)
        self.client.post(
            "/v1/query",
            json={"question": question, "top_k": 5},
            headers={"X-API-Key": API_KEY},
        )

    @task(3)
    def query_with_session(self) -> None:
        question = random.choice(QUESTIONS)
        resp = self.client.post(
            "/v1/query",
            json={
                "question": question,
                "top_k": 5,
                "session_id": "load-test-session",
            },
            headers={"X-API-Key": API_KEY},
        )
        if resp.status_code == 200:
            data = resp.json()
            session_id = data.get("session_id")
            if session_id:
                followup = random.choice(QUESTIONS)
                self.client.post(
                    "/v1/query",
                    json={
                        "question": followup,
                        "top_k": 5,
                        "session_id": session_id,
                    },
                    headers={"X-API-Key": API_KEY},
                )

    @task(1)
    def query_varied_top_k(self) -> None:
        question = random.choice(QUESTIONS)
        top_k = random.choice([3, 5, 10, 15])
        self.client.post(
            "/v1/query",
            json={"question": question, "top_k": top_k},
            headers={"X-API-Key": API_KEY},
        )


class RAGHealthUser(HttpUser):
    """Simulates monitoring/health check traffic."""

    wait_time = between(1.0, 5.0)
    weight = 1  # 10% of traffic is health checks

    @task
    def health_check(self) -> None:
        self.client.get("/v1/health")

    @task
    def metrics(self) -> None:
        self.client.get("/metrics")


class RAGIngestUser(HttpUser):
    """Simulates occasional document ingestion."""

    wait_time = between(5.0, 15.0)
    weight = 1  # 10% of traffic is ingestion

    @task
    def ingest_small_file(self) -> None:
        content = f"Load test document content {random.randint(1, 100000)}."
        self.client.post(
            "/v1/ingest",
            files=[("files", ("test.txt", content.encode(), "text/plain"))],
            headers={"X-API-Key": API_KEY},
        )
