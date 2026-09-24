#!/usr/bin/env python3
"""
Unit tests for Grant 1FAB0 Connectome Evaluation Viewer (src/viewer.py).
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.request
import urllib.error

from src.viewer import parse_args, create_handler, ViewerRequestHandler, DEFAULT_PORT, DEFAULT_HOST
import http.server


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestViewerArgs(unittest.TestCase):
    def test_default_args(self):
        args = parse_args([])
        self.assertEqual(args.port, DEFAULT_PORT)
        self.assertEqual(args.host, DEFAULT_HOST)
        self.assertFalse(args.no_browser)
        self.assertEqual(args.docs_dir, "docs")
        self.assertEqual(args.results_dir, "results")
        self.assertEqual(args.battery_dir, "battery")

    def test_custom_args(self):
        args = parse_args([
            "--port", "9095",
            "--host", "0.0.0.0",
            "--no-browser",
            "--docs-dir", "custom_docs",
            "--results-dir", "custom_results",
            "--battery-dir", "custom_battery"
        ])
        self.assertEqual(args.port, 9095)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertTrue(args.no_browser)
        self.assertEqual(args.docs_dir, "custom_docs")
        self.assertEqual(args.results_dir, "custom_results")
        self.assertEqual(args.battery_dir, "custom_battery")


class TestViewerPathTranslation(unittest.TestCase):
    def setUp(self):
        self.docs_dir = os.path.abspath("docs")
        self.results_dir = os.path.abspath("results")
        self.battery_dir = os.path.abspath("battery")
        self.handler_cls = create_handler(self.docs_dir, self.results_dir, self.battery_dir)

    def test_translate_results_path(self):
        handler = self.handler_cls.__new__(self.handler_cls)
        handler.docs_dir = self.docs_dir
        handler.results_dir = self.results_dir
        handler.battery_dir = self.battery_dir

        target = handler.translate_path("/results/smoke-v4.jsonl")
        expected = os.path.join(self.results_dir, "smoke-v4.jsonl")
        self.assertEqual(target, expected)

    def test_translate_battery_path(self):
        handler = self.handler_cls.__new__(self.handler_cls)
        handler.docs_dir = self.docs_dir
        handler.results_dir = self.results_dir
        handler.battery_dir = self.battery_dir

        target = handler.translate_path("/battery/battery-v4.json")
        expected = os.path.join(self.battery_dir, "battery-v4.json")
        self.assertEqual(target, expected)

    def test_translate_docs_path(self):
        handler = self.handler_cls.__new__(self.handler_cls)
        handler.docs_dir = self.docs_dir
        handler.results_dir = self.results_dir
        handler.battery_dir = self.battery_dir

        target = handler.translate_path("/BATTERY.md")
        expected = os.path.join(self.docs_dir, "BATTERY.md")
        self.assertEqual(target, expected)

    def test_traversal_protection(self):
        handler = self.handler_cls.__new__(self.handler_cls)
        handler.docs_dir = self.docs_dir
        handler.results_dir = self.results_dir
        handler.battery_dir = self.battery_dir

        target = handler.translate_path("/results/../../etc/passwd")
        self.assertTrue(target.endswith("__nonexistent__"))


class TestViewerServerLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.host = "127.0.0.1"
        cls.docs_dir = os.path.abspath("docs")
        cls.results_dir = os.path.abspath("results")
        cls.battery_dir = os.path.abspath("battery")
        cls.handler_cls = create_handler(cls.docs_dir, cls.results_dir, cls.battery_dir)
        cls.httpd = http.server.ThreadingHTTPServer((cls.host, cls.port), cls.handler_cls)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_root_serves_index(self):
        url = f"http://{self.host}:{self.port}/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            html = resp.read().decode("utf-8")
            self.assertIn("Grant 1FAB0: Fly Connectome Behavioral Evaluation Arena", html)
            self.assertIn("Dual-Decoder", html)

    def test_api_results_endpoint(self):
        url = f"http://{self.host}:{self.port}/api/results"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(data, list)
            names = [item["name"] for item in data]
            self.assertIn("smoke-v4.jsonl", names)

    def test_results_file_fetch(self):
        url = f"http://{self.host}:{self.port}/results/smoke-v4.jsonl"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/x-ndjson", resp.headers.get("Content-Type", ""))
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
            content = resp.read().decode("utf-8")
            self.assertIn("battery-v4.json", content)

    def test_battery_file_fetch(self):
        url = f"http://{self.host}:{self.port}/battery/battery-v4.json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("items", data)

    def test_options_cors(self):
        url = f"http://{self.host}:{self.port}/api/results"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")

    def test_nonexistent_returns_404(self):
        url = f"http://{self.host}:{self.port}/results/__definitely_not_exist__.jsonl"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url)
        self.assertEqual(ctx.exception.code, 404)


class TestEmbeddedBenchmarkProvenance(unittest.TestCase):
    def setUp(self):
        index_path = os.path.join(os.path.dirname(__file__), "docs", "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        m = re.search(r'const EMBEDDED_BENCHMARK = (\{.*?\});', content)
        self.assertIsNotNone(m, "EMBEDDED_BENCHMARK must exist in docs/index.html")
        self.benchmark = json.loads(m.group(1))

    def test_embedded_metadata(self):
        self.assertEqual(self.benchmark.get("battery"), "battery-v4")
        self.assertEqual(
            self.benchmark.get("battery_sha256"),
            "e90b093bbbd7898b726cf4cc41167b3f7d010c888cd47d3e4a007e25f6392991"
        )
        sample_rows = self.benchmark.get("sample_rows", [])
        self.assertEqual(len(sample_rows), 36)

    def test_embedded_rows_verbatim_provenance(self):
        sample_rows = self.benchmark.get("sample_rows", [])
        for row in sample_rows:
            self.assertNotIn(
                "trajectory", row,
                f"Row {row.get('item')} {row.get('condition')} must not have trajectory key; rows must be verbatim from runs-v4.jsonl"
            )
            self.assertIn("sha256", row)
            self.assertIn("prev", row)
            self.assertIn("battery_sha256", row)

        # Check against results/runs-v4.jsonl if available locally or via git pr-6
        v4_content = None
        runs_v4_path = os.path.join(os.path.dirname(__file__), "results", "runs-v4.jsonl")
        if os.path.exists(runs_v4_path):
            with open(runs_v4_path, "r", encoding="utf-8") as f:
                v4_content = f.read()
        else:
            try:
                import subprocess
                v4_content = subprocess.check_output(
                    ["git", "show", "pr-6:results/runs-v4.jsonl"],
                    stderr=subprocess.DEVNULL
                ).decode("utf-8")
            except Exception:
                pass

        if v4_content:
            for row in sample_rows:
                self.assertIn(
                    row["sha256"], v4_content,
                    f"Row sha256 {row['sha256']} must be found verbatim in results/runs-v4.jsonl"
                )

    def test_embedded_verdicts(self):
        verdicts = self.benchmark.get("verdicts", {})
        self.assertEqual(len(verdicts), 6)
        self.assertEqual(verdicts["1"]["verdict"], "failed")
        self.assertEqual(verdicts["2"]["verdict"], "failed")
        self.assertEqual(verdicts["3"]["verdict"], "held")
        self.assertEqual(verdicts["4"]["verdict"], "held")
        self.assertEqual(verdicts["5"]["verdict"], "held")
        self.assertEqual(verdicts["6"]["verdict"], "failed")


if __name__ == "__main__":
    unittest.main()

