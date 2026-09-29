"""Check an LLM's answer before it reaches a user, in milliseconds (Jev's "LLM verification" use case).

  pip install -e sdk/python
  python examples/verify_llm_output.py --url http://localhost:8000 --api-key <key>

Dragonfly reads the source and the LLM's answer once and answers four yes/no checks in one pass. With max_error it
only blocks or passes when it is sure at that error rate; everything else goes to a person.
"""

import argparse

from dragonfly_client import Dragonfly

CHECKS = {
    "grounded": "Is every claim in the answer supported by the source text?",
    "on_policy": "Does the answer follow the policy (no promises of refunds, no legal or medical advice)?",
    "leaks_data": "Does the answer reveal another customer's personal data?",
}


def verify(df: Dragonfly, source: str, answer: str, max_error: float | None = None) -> str:
    d = df.decide({"source": source, "llm_answer": answer}, instructions=CHECKS, max_error=max_error,
                  grounded=bool, on_policy=bool, leaks_data=bool)
    sure = (lambda x: x.decided) if max_error else (lambda x: x.confidence >= 0.6)
    if (d.leaks_data and sure(d.leaks_data)) or (not d.grounded and sure(d.grounded)) or \
            (not d.on_policy and sure(d.on_policy)):
        return "block"
    if all(sure(x) for x in d.values()):
        return "send"
    return "review"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--max-error", type=float)
    a = ap.parse_args()
    source = "Order 5512 was shipped on May 3 and delivered on May 6. Refunds require a return within 30 days."
    with Dragonfly(a.url, api_key=a.api_key) as df:
        for answer in ("Your order 5512 was delivered on May 6.",
                       "Your order 5512 was delivered on May 6. I have issued a full refund to your card.",
                       "Order 5512 went to Jane Doe, 12 Elm St, phone 555-0199."):
            print(f"{verify(df, source, answer, a.max_error):>6}  {answer}")


if __name__ == "__main__":
    main()
