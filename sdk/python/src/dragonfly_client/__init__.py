"""Dragonfly client: typed decisions from Python types.

    from typing import Literal
    from dragonfly_client import Dragonfly, Scale

    df = Dragonfly("http://localhost:8000", api_key="...")
    d = df.decide(ticket_text,
                  urgent=bool,                                   # yes/no
                  intent=Literal["refund", "delivery", "other"], # one of these
                  mood=Scale(["angry", "neutral", "happy"]),    # ordered levels
                  instructions={"urgent": "Does this need a human agent right away?"})
    if d.urgent:            # a Decision is truthy for yes/no; .value, .confidence, .probabilities
        escalate(d.intent.value)

    @df.decision
    def needs_refund(ticket: str) -> bool:
        "Does the customer ask for their money back?"

    if needs_refund(text): ...      # a smart if-statement

Works with any server that speaks the TypeSafe System One API; Dragonfly-only features (max_error, streaming,
batch, specialists) are optional arguments.
"""

from .client import AsyncDragonfly, Decision, Decisions, Dragonfly, DragonflyError, Scale, question

__all__ = ["AsyncDragonfly", "Decision", "Decisions", "Dragonfly", "DragonflyError", "Scale", "question"]
__version__ = "0.1.0"
