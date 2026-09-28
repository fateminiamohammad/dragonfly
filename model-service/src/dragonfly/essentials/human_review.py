"""human-review: send unsure decisions to people, and turn their answers into training data (active learning).

For every answer below DRAGONFLY_LOW_CONFIDENCE, the document, the question and the model's probabilities go to a
review queue in Redis. The model's answer is still returned immediately (this plugin never replaces it; put
llm-escalation after it to also escalate). Reviewers label items in the UI's Review page; the backend exports labelled
items as Dragonfly training JSONL, ready for dragonfly-train or a specialist.

  DRAGONFLY_PLUGINS=human-review
  DRAGONFLY_LOW_CONFIDENCE=0.3
  DRAGONFLY_PLUGIN_HUMAN_REVIEW_REDIS_URL=redis://redis:6379     (default: REDIS_URL)
  DRAGONFLY_PLUGIN_HUMAN_REVIEW_SAMPLE=1.0                       (fraction of unsure answers to queue)
  DRAGONFLY_PLUGIN_HUMAN_REVIEW_MAX_PENDING=10000                (older pending items are trimmed)
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid

from ..plugins import Plugin, PluginContext
from ..schema import DecideRequest

PENDING = "dragonfly:review:pending"


def review_item(request: DecideRequest, question_id: str, answer: dict) -> dict:
    q = request.questions[question_id]
    return {
        "id": uuid.uuid4().hex,
        "created": time.time(),
        "model": request.model,
        "state": request.state,
        "question_id": question_id,
        "question": {"type": q.type, "instructions": q.instructions, "criteria": q.criteria},
        "answer": answer,
    }


class HumanReview(Plugin):
    name = "human-review"
    version = "1.0.0"
    api_version = "1.1"
    blocking = True  # a Redis round trip
    timeout_s = 1.0
    fail_open = True  # a Redis hiccup must not fail the decision

    def setup(self, ctx: PluginContext) -> None:
        import redis

        url = ctx.config.get("redis_url") or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.Redis.from_url(url, socket_timeout=1)
        self.sample = float(ctx.config.get("sample", "1.0"))
        self.max_pending = int(ctx.config.get("max_pending", "10000"))

    def on_low_confidence(self, request: DecideRequest, question_id: str, answer: dict) -> dict | None:
        if random.random() < self.sample:
            pipe = self.redis.pipeline()
            pipe.lpush(PENDING, json.dumps(review_item(request, question_id, answer), default=str))
            pipe.ltrim(PENDING, 0, self.max_pending - 1)
            pipe.execute()
        return None  # keep the model's answer; the review happens later
