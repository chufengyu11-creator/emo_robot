#!/usr/bin/env python3

"""Send one prompt JSON request to the computer relay and print the response."""

from __future__ import annotations

import argparse
import json
import socket
import time
import uuid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one prompt_request JSON to tcp_json_prompt_relay.py.",
    )
    parser.add_argument(
        "relay_host",
        help="Computer relay host reachable from the robot, e.g. 10.0.1.6.",
    )
    parser.add_argument("--relay-port", type=int, default=8766)
    parser.add_argument(
        "--prompt",
        default="请用一句话回复：机器人到服务器的请求响应链路已经通了吗？",
    )
    parser.add_argument("--timeout-s", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = {
        "type": "prompt_request",
        "request_id": str(uuid.uuid4()),
        "prompt": args.prompt,
        "sent_at": time.time(),
        "source": "robot_prompt_client",
    }
    line = json.dumps(request, ensure_ascii=False) + "\n"
    with socket.create_connection(
        (args.relay_host, args.relay_port),
        timeout=args.timeout_s,
    ) as sock:
        sock.sendall(line.encode("utf-8"))
        with sock.makefile("r", encoding="utf-8") as stream:
            response_line = stream.readline()
    if not response_line:
        raise RuntimeError("relay closed without a response")
    response = json.loads(response_line)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
