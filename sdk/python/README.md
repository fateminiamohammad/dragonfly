# dragonfly-client

Python client for [Dragonfly](https://github.com/fateminiamohammad/dragonfly). It also works with any server that speaks the
TypeSafe System One API. You write questions as Python types and get typed, calibrated decisions back in
milliseconds.

```bash
pip install -e sdk/python      # from the Dragonfly repository
```

```python
from typing import Literal
from dragonfly_client import Dragonfly, Scale

df = Dragonfly("http://localhost:8000", api_key="df_...")

d = df.decide(ticket,
              urgent=bool,                                    # yes / no
              intent=Literal["refund", "delivery", "other"],  # one of these
              mood=Scale(["angry", "neutral", "happy"]),     # ordered levels
              instructions={"urgent": "Does this need a human agent right away?"})

if d.urgent:                          # yes/no decisions are truthy
    route(d.intent.value, confidence=d.intent.confidence)
```

## A smart if-statement

```python
@df.decision
def wants_refund(ticket: str) -> bool:
    "Does the customer ask for their money back?"

if wants_refund(text):
    ...
```

The docstring is the question and the return annotation is its type. `Literal[...]`, an `Enum`, a list or a dict of
name → description make a choice. `Annotated[float, Scale([...])]` makes a score.

## More

| | |
|---|---|
| `df.batch(states, urgent=bool)` | the same questions over many documents in one call (`/v1/batch`) |
| `for event, d in df.stream(state, urgent=bool)` | anytime answers: `answers` (tier S, ms), `update` (tier M), `done` |
| `df.decide(..., max_error=0.05)` | each decision gets `.decided` (within the error rate) and `.set` (the options containing the truth) |
| `Dragonfly(..., model="invoices")` / `model="auto"` | ask a specialist, or let Dragonfly route |
| `AsyncDragonfly` | the same with `await` / `async for` |

A decision has `.value` (True/False, the chosen name, or the expected level), `.confidence`, `.probabilities`,
`.tier` and `.raw`. `Decisions` (the result of `decide`) gives each answer by name, plus `.latency_ms`,
`.specialist` and `.response`.
