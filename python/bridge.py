"""
Client side of the bridge to the headless Godot host.

One JSON object per line in both directions. 
Newline marks where each message ends.
Python sends one message and waits for one reply.
Godot never speaks first so simulation only moves when trainer asks.
(so training is deterministic and never runs ahead)
Every message has a "type", and Godot answers anything that it can't handle
with {"type": "error", "reason": ...} (instead of staying silent).
Godot listens and Python connects on 127.0.0.1:11008 (so Python can launch Godot as a subprocess)
"""

import json
import socket

class Bridge:
    def __init__(self, host="127.0.0.1", port=11008, timeout=30.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = self._sock.makefile("r", encoding="utf-8", newline="\n")

    def request(self, message: dict) -> dict:
        self._sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
        line = self._reader.readline()
        if not line:
            raise ConnectionError("host closed the connection")
        reply = json.loads(line)
        if reply.get("type") == "error":
            raise RuntimeError(f"host error: {reply.get('reason')}")
        return reply

    def close(self):
        self._reader.close()
        self._sock.close()


if __name__ == "__main__":
    bridge = Bridge()
    print(bridge.request({"type": "hello", "seed": 7, "ships": 32, "repeat": 6}))
    bridge.close()