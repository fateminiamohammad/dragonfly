"""End-to-end check through nginx: log in, create an API key, decide on an image and a voice clip, convert media to text.

  python scripts/e2e_media.py --url http://localhost:8080 --email admin@example.com --password <ADMIN_PASSWORD>
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "runs" / "samples"


def call(url, body=None, token=None, method=None):
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers,
                                 method=method or ("POST" if body is not None else "GET"))
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()), (time.perf_counter() - t) * 1000


def b64(name):
    return base64.b64encode((SAMPLES / name).read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--email", default="admin@example.com")
    ap.add_argument("--password", required=True)
    a = ap.parse_args()

    jwt = call(f"{a.url}/api/auth/login", {"email": a.email, "password": a.password})[0]["token"]
    key = call(f"{a.url}/api/keys", {"name": "e2e media"}, jwt)[0]
    print("created API key", key["prefix"] + "…")
    try:
        request = {
            "state": {"note": "customer upload", "document": {"type": "image", "data": b64("invoice.png"), "name": "invoice.png"},
                      "voice_message": {"type": "audio", "data": b64("speech.flac"), "language": "en"}},
            "questions": {
                "doc_type": {"type": "choice", "instructions": "What kind of document was uploaded?",
                             "criteria": {"invoice": None, "passport": None, "contract": None, "photo": None}},
                "amount": {"type": "choice", "instructions": "What is the total amount due?",
                           "criteria": {"$1,250.00": None, "$4,821.00": None, "$2,026.00": None}},
                "food": {"type": "noul", "instructions": "Does the voice message talk about food?"},
            },
        }
        out, ms = call(f"{a.url}/v1/systemone", request, key["key"])
        print(f"/v1/systemone with image + audio: {ms:.0f} ms round trip "
              f"(media {out['usage']['media_ms']:.0f} ms, model {out['latency_ms']:.1f} ms)")
        for qid, ans in out["answers"].items():
            value = ans.get("choice") if ans["type"] == "choice" else ans.get("noul")
            print(f"  {qid:9} -> {value}  (confidence {ans['confidence']:.2f}, tier {ans.get('tier')})")
        checks = {"doc_type": out["answers"]["doc_type"]["choice"] == "invoice",
                  "amount": out["answers"]["amount"]["choice"] == "$1,250.00",
                  "food": out["answers"]["food"]["noul"] > 0.5}
        conv, ms = call(f"{a.url}/v1/perceive", {"items": [
            {"id": "img", "type": "image", "data": b64("cats.jpg")},
            {"id": "voice", "type": "audio", "data": b64("speech.flac"), "language": "en"}]}, key["key"])
        print(f"/v1/perceive: {ms:.0f} ms | cats.jpg shows {[t['label'] for t in conv['items']['img']['tags'][:3]]}"
              f" | transcript: {conv['items']['voice']['transcript'][:60]}…")
        checks["tags"] = conv["items"]["img"]["tags"][0]["label"] == "cat"
        checks["transcript"] = conv["items"]["voice"]["transcript"].startswith("he hoped there would be stew")
        print("checks:", checks)
        if not all(checks.values()):
            raise SystemExit("some end-to-end checks failed")
        print("all end-to-end media checks passed")
    finally:
        call(f"{a.url}/api/keys/{key['id']}", None, jwt, method="DELETE")
        print("revoked the test key")


if __name__ == "__main__":
    main()
