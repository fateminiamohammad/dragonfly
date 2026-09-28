"""Free plugins that ship with Dragonfly. Enable any of them by name in DRAGONFLY_PLUGINS:

  llm-escalation   unsure answers are re-asked to an LLM (local Ollama or any OpenAI-compatible API)
  webhook-audit    send decisions to a webhook, Slack or a JSONL file, in the background
  guardrails       PII redaction, prompt-injection and profanity checks, state size limit
  human-review     unsure decisions go to a review queue; reviewed answers become training data

Settings are DRAGONFLY_PLUGIN_<NAME>_<KEY>, documented in each module.
"""
