"""Example out-of-process Dragonfly plugin: a policy sidecar, written the way any language would do it (stubs
generated from proto/dragonfly/plugin/v1/plugin.proto by protoc).

Rules:
  on_request   reject states longer than MAX_STATE_CHARS with HTTP 413
  on_decision  flag the response with "review_required": true when any answer's confidence is below MIN_CONFIDENCE

  MAX_STATE_CHARS=20000 MIN_CONFIDENCE=0.5 python server.py   # listens on :50051
"""

import json
import os
from concurrent import futures

import grpc

from dragonfly.plugin.v1 import plugin_pb2 as pb
from dragonfly.plugin.v1 import plugin_pb2_grpc as rpc

MAX_STATE_CHARS = int(os.environ.get("MAX_STATE_CHARS", "20000"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.5"))


class Policy(rpc.PluginServiceServicer):
    def Describe(self, request, context):
        return pb.PluginInfo(name="policy", version="0.1.0", api_version="1.0", hooks=["on_request", "on_decision"])

    def OnRequest(self, request, context):
        state = json.loads(request.request_json)["state"]
        size = len(state if isinstance(state, str) else json.dumps(state))
        if size > MAX_STATE_CHARS:
            return pb.HookReply(reject_status=413, reject_message=f"state has {size} characters; the limit is {MAX_STATE_CHARS}")
        return pb.HookReply()  # unchanged

    def OnDecision(self, request, context):
        response = json.loads(request.response_json)
        low = [qid for qid, a in response["answers"].items() if a.get("confidence", 1) < MIN_CONFIDENCE]
        response["review_required"] = bool(low)
        response["low_confidence_questions"] = low
        return pb.HookReply(response_json=json.dumps(response))

    def OnLowConfidence(self, request, context):
        return pb.HookReply()


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    rpc.add_PluginServiceServicer_to_server(Policy(), server)
    server.add_insecure_port(f"[::]:{os.environ.get('PORT', '50051')}")
    server.start()
    print("policy plugin listening", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
