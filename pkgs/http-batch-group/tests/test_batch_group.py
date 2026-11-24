import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "qwen38-batch-group.py"
WORKLOAD = ROOT / "workloads/qwen38-subagent-workload.json"


class _State:
    def __init__(self, fail_first_n: int = 0):
        self.lock = threading.Lock()
        self.fail_first_n = fail_first_n
        self.requests: list[dict] = []


def _make_handler(state: _State):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            with state.lock:
                index = len(state.requests)
                state.requests.append(body)
                should_fail = index < state.fail_first_n
            if self.path != "/v1/completions":
                self.send_response(404)
                self.end_headers()
                return
            if should_fail:
                payload = json.dumps({"error": "injected group miss"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = json.dumps(
                {
                    "id": f"cmpl-{index}",
                    "choices": [{"index": 0, "text": f"reply:{body.get('prompt', '')}"}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    return Handler


class HttpBatchGroupTests(unittest.TestCase):
    def _start_server(self, fail_first_n: int = 0):
        state = _State(fail_first_n=fail_first_n)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _make_handler(state)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]
        return f"http://127.0.0.1:{port}", state

    def test_barrier_release_writes_ordered_responses(self):
        base_url, state = self._start_server()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reqs = []
            for i in range(4):
                p = root / f"req-{i}.json"
                p.write_text(json.dumps({"model": "test", "prompt": f"p{i}"}))
                reqs.append(p)
            out_dir = root / "out"
            cmd = [
                sys.executable,
                "-I",
                str(DRIVER),
                "--url",
                base_url,
                "--output-directory",
                str(out_dir),
                "--attempts",
                "1",
            ]
            for p in reqs:
                cmd.extend(["--request", str(p)])
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertIn("4/4 usable responses", proc.stdout)
            self.assertEqual(len(state.requests), 4)
            for i in range(4):
                resp = json.loads((out_dir / f"response-{i}.json").read_text())
                self.assertEqual(resp["choices"][0]["text"], f"reply:p{i}")

    def test_retries_on_http_error_and_rejects_https_scheme(self):
        base_url, state = self._start_server(fail_first_n=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = root / "req-0.json"
            req.write_text(json.dumps({"model": "test", "prompt": "retry-me"}))
            out_dir = root / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(DRIVER),
                    "--url",
                    base_url,
                    "--request",
                    str(req),
                    "--output-directory",
                    str(out_dir),
                    "--attempts",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("group attempt 1: 0/1 usable responses", proc.stdout)
            self.assertIn("group attempt 2: 1/1 usable responses", proc.stdout)
            self.assertEqual(len(state.requests), 2)

            bad = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(DRIVER),
                    "--url",
                    "https://127.0.0.1:8443",
                    "--request",
                    str(req),
                    "--output-directory",
                    str(out_dir),
                    "--attempts",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("only http is supported", bad.stderr)

    def test_frozen_subagent_workload_schema(self):
        data = json.loads(WORKLOAD.read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["workload_id"], "qwen38-three-review-subagents-v1")
        self.assertEqual(
            [w["name"] for w in data["workers"]],
            ["BoundarySearch", "LoginQuery", "FanoutCapacity"],
        )
        for worker in data["workers"]:
            self.assertTrue(worker["task"])
            self.assertTrue(worker["required_terms"])


if __name__ == "__main__":
    unittest.main()
