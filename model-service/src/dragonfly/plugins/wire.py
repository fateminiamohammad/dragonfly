"""Minimal protobuf wire codec for the messages in proto/dragonfly/plugin/v1/plugin.proto.

Those messages only use string, bool and int32 fields, which lets the host speak the protocol without generated code
or a protoc build step. Plugins in other languages use normal protoc-generated stubs; the bytes are identical.
"""

from __future__ import annotations

# field number -> (name, kind); kind: "str", "strs" (repeated string), "bool", "int"
SCHEMAS: dict[str, dict[int, tuple[str, str]]] = {
    "DescribeRequest": {1: ("host_api_version", "str")},
    "PluginInfo": {1: ("name", "str"), 2: ("version", "str"), 3: ("api_version", "str"), 4: ("hooks", "strs")},
    "HookRequest": {1: ("request_json", "str"), 2: ("response_json", "str"), 3: ("question_id", "str"),
                    4: ("answer_json", "str")},
    "HookReply": {1: ("request_json", "str"), 2: ("response_json", "str"), 3: ("answer_json", "str"),
                  4: ("replaced", "bool"), 5: ("reject_status", "int"), 6: ("reject_message", "str")},
}


def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64  # proto3 int32 negatives are 10-byte varints
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def encode(message: str, values: dict) -> bytes:
    out = bytearray()
    for num, (name, kind) in SCHEMAS[message].items():
        v = values.get(name)
        if v in (None, "", False, 0, []):
            continue  # proto3 defaults are not serialized
        items = v if kind == "strs" else [v]
        for item in items:
            if kind in ("str", "strs"):
                data = str(item).encode()
                out += _varint(num << 3 | 2) + _varint(len(data)) + data
            else:
                out += _varint(num << 3 | 0) + _varint(int(item))
    return bytes(out)


def decode(message: str, buf: bytes) -> dict:
    schema = SCHEMAS[message]
    values: dict = {name: ([] if kind == "strs" else "" if kind == "str" else False if kind == "bool" else 0)
                    for name, kind in schema.values()}
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 2:
            length, i = _read_varint(buf, i)
            raw, i = buf[i:i + length], i + length
            if num in schema:
                name, kind = schema[num]
                if kind == "strs":
                    values[name].append(raw.decode())
                else:
                    values[name] = raw.decode()
        elif wire == 0:
            n, i = _read_varint(buf, i)
            if num in schema:
                name, kind = schema[num]
                if n >= 1 << 63:
                    n -= 1 << 64
                values[name] = bool(n) if kind == "bool" else n
        elif wire == 5:
            i += 4  # unknown fixed32
        elif wire == 1:
            i += 8  # unknown fixed64
        else:
            raise ValueError(f"unsupported wire type {wire} in {message}")
    return values
