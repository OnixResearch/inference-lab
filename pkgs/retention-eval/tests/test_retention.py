import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import retention as r

try:
    import torch
except ImportError:
    torch = None

REQUIRE_TORCH = os.environ.get("REQUIRE_TORCH") == "1"


class CheckpointContentTests(unittest.TestCase):
    def test_index_and_every_unique_referenced_shard_are_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r.save(
                root / "model.safetensors.index.json",
                {
                    "weight_map": {
                        "a": "first.safetensors",
                        "b": "second.safetensors",
                        "c": "second.safetensors",
                    }
                },
            )
            (root / "first.safetensors").write_bytes(b"first weights")
            (root / "second.safetensors").write_bytes(b"second weights")
            manifest = r.weight_manifest(root)
            self.assertEqual(
                set(manifest["shards"]), {"first.safetensors", "second.safetensors"}
            )
            self.assertEqual(
                manifest["index"]["sha256"],
                r.digest(root / "model.safetensors.index.json"),
            )
            self.assertEqual(
                manifest["shards"]["first.safetensors"]["sha256"],
                r.digest(root / "first.safetensors"),
            )
            self.assertEqual(
                manifest["shards"]["second.safetensors"]["sha256"],
                r.digest(root / "second.safetensors"),
            )
            (root / "second.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                r.weight_manifest(root)

    def test_index_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r.save(
                root / "model.safetensors.index.json",
                {"weight_map": {"a": "../outside.safetensors"}},
            )
            with self.assertRaisesRegex(ValueError, "inside the model"):
                r.weight_manifest(root)


class RetentionStdlibTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prov = {
            "model": "synthetic",
            "checkpoint": "test",
            "runtime": "cpu",
            "package": "baseline-pkg",
            "cache": "none",
            "precision_flags": {"MODE": "0"},
            "execution": "synthetic-cpu-not-model",
        }

    def _receipt(self, values, *, prov=None, baseline_sha256=None, declared_changes=None):
        return {
            "schema": r.SCHEMA,
            "recipe": r.RECIPE,
            "corpus_sha256": "a" * 64,
            "logits_sha256": "b" * 64,
            "provenance": dict(prov or self.prov),
            "tokenizer_assets_sha256": {"tokenizer.json": "c" * 64},
            "vocabulary_size": 248320,
            "model_config_sha256": "d" * 64,
            "scored_token_counts": [76, 122, 50, 46, 42, 49],
            "passage_nll": values,
            "mean_nll": sum(values) / 6,
            "scoring_engine": "synthetic:cpu",
            "baseline_sha256": baseline_sha256,
            "declared_changes": declared_changes or {},
        }

    def test_shipped_frozen_corpus_validates(self):
        frozen = r.corpus(ROOT / "data/qwen38-six-passages.json")
        self.assertEqual(frozen["vocabulary_size"], 248320)
        self.assertEqual(
            [len(p["input_ids"]) for p in frozen["passages"]],
            [89, 135, 63, 59, 55, 62],
        )
        self.assertEqual(
            [len(p["target_indices"]) for p in frozen["passages"]],
            [76, 122, 50, 46, 42, 49],
        )

    def test_compare_and_cli_enforce_content_binding_and_diagnostic_gate(self):
        base_path = self.root / "base.json"
        trial_path = self.root / "trial.json"
        out_path = self.root / "cmp.json"
        r.save(base_path, self._receipt([0.7, 0.8, 0.6, 0.9, 0.75, 0.72]))
        changes = {"precision_flags": {"MODE": "1"}}
        r.save(
            trial_path,
            self._receipt(
                [0.705, 0.805, 0.605, 0.905, 0.755, 0.725],
                prov=dict(self.prov, **changes),
                baseline_sha256=r.digest(base_path),
                declared_changes=changes,
            ),
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "cli.py"),
                "compare",
                "--baseline",
                str(base_path),
                "--treatment",
                str(trial_path),
                "--original-diagnostic",
                "--out",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(proc.stdout)
        self.assertAlmostEqual(parsed["mean_nll_delta"], 0.005, places=12)
        self.assertTrue(parsed["original_diagnostic"]["passes"])
        self.assertEqual(json.loads(out_path.read_text()), parsed)

        # Mutate baseline after trial binding -> rejected by CLI with exit code 2.
        tampered = r.load(base_path)
        tampered["provenance"]["runtime"] = "other"
        r.save(base_path, tampered)
        bad = subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "cli.py"),
                "compare",
                "--baseline",
                str(base_path),
                "--treatment",
                str(trial_path),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(bad.returncode, 2)
        self.assertIn("content-bound", bad.stderr)

    def test_hardware_receipt_without_weight_manifest_rejected(self):
        prov = dict(self.prov, execution="hardware:synthetic-test-fixture")
        with self.assertRaisesRegex(ValueError, "all-shard content"):
            r.validate_receipt(self._receipt([0.5] * 6, prov=prov))

    def test_undeclared_or_noop_treatment_changes_rejected(self):
        with self.assertRaisesRegex(ValueError, "only declared"):
            r.matched(self.prov, dict(self.prov, runtime="other"), {"runtime": "other"})
        with self.assertRaisesRegex(ValueError, "does not change provenance"):
            r.matched(self.prov, dict(self.prov), {"package": "baseline-pkg"})


@unittest.skipIf(
    torch is None and not REQUIRE_TORCH,
    "torch not installed in stdlib-only retention-eval check",
)
class RetentionTorchTests(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.fail("REQUIRE_TORCH=1 but torch is unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frozen = r.load(r.ROOT / "data/qwen38-six-passages.json")
        self.frozen["vocabulary_size"] = 8
        self.frozen["tokenizer_assets_sha256"] = {"synthetic": "CPU-test-not-model"}
        for i, p in enumerate(self.frozen["passages"]):
            p["input_ids"] = [(j + i) % 8 for j in range(14 + i * 3)]
            p["target_indices"] = list(range(13, len(p["input_ids"])))
        self.corpus = self.root / "corpus.json"
        r.save(self.corpus, self.frozen)
        self.prov = {
            "model": "synthetic",
            "checkpoint": "test",
            "runtime": "cpu",
            "package": "test",
            "cache": "none",
            "precision_flags": {},
            "execution": "synthetic-cpu-not-model",
        }
        self.export = {
            "corpus_sha256": r.digest(self.corpus),
            "provenance": self.prov,
            "target_indices": [p["target_indices"] for p in self.frozen["passages"]],
            "rows": [],
        }
        for i, p in enumerate(self.frozen["passages"]):
            logits = torch.zeros((len(p["target_indices"]), 8), dtype=torch.float64)
            for j, target in enumerate(p["input_ids"][13:]):
                logits[j, target] = i + 0.123456789
            self.export["rows"].append(logits)
        self.logits = self.root / "logits.pt"
        self.base = self.root / "base.json"

    def run_score(self, **kwargs):
        torch.save(self.export, self.logits)
        return r.score(self.corpus, self.logits, self.base, **kwargs)

    def test_f32_then_double_equal_passage_weighting(self):
        out = self.run_score()
        expected = [
            math.log(
                math.exp(float(torch.tensor(i + 0.123456789, dtype=torch.float32))) + 7
            )
            - float(torch.tensor(i + 0.123456789, dtype=torch.float32))
            for i in range(6)
        ]
        for actual, want in zip(out["passage_nll"], expected):
            self.assertAlmostEqual(actual, want, places=13)
        self.assertAlmostEqual(out["mean_nll"], sum(expected) / 6, places=13)
        pooled = sum(v * n for v, n in zip(expected, out["scored_token_counts"])) / sum(
            out["scored_token_counts"]
        )
        self.assertGreater(abs(out["mean_nll"] - pooled), 0.1)

    def test_target_offbyone_and_padding_and_nonfinite_rejected(self):
        self.export["target_indices"][0] = [12]
        with self.assertRaisesRegex(ValueError, "13..end"):
            self.run_score()
        self.export["target_indices"][0] = self.frozen["passages"][0]["target_indices"]
        original = self.export["rows"][0]
        for bad in (
            original[:0],
            torch.full_like(original, float("nan")),
            original[:, :3],
            torch.zeros((len(original), 9)),
        ):
            self.export["rows"][0] = bad
            with self.assertRaises(ValueError):
                self.run_score()

    def test_matched_receipts_and_weight_change_refusal(self):
        weights = self.root / "weights"
        weights.mkdir()
        r.save(
            weights / "model.safetensors.index.json",
            {"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}},
        )
        (weights / "first.safetensors").write_bytes(b"same first shard")
        second = weights / "second.safetensors"
        second.write_bytes(b"original second shard")
        original = r.weight_manifest(weights)
        self.prov["model"] = r.weight_identity(original)
        self.prov["execution"] = "hardware:synthetic-test-fixture"
        self.export["weight_manifest"] = original
        self.run_score()
        second.write_bytes(b"different second shard")
        changed = r.weight_manifest(weights)
        self.export["weight_manifest"] = changed
        self.export["provenance"] = dict(self.prov, model=r.weight_identity(changed))
        torch.save(self.export, self.logits)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            r.score(self.corpus, self.logits, self.root / "trial.json", self.base, {})


if __name__ == "__main__":
    unittest.main()
