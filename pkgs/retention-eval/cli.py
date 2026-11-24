"""Standalone backend-agnostic CLI for provenance-bound six-passage retention."""

import argparse
import json
import sys
from pathlib import Path

# The installed entry point uses Python isolated mode; only our own module directory is added.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retention


def parser():
    p = argparse.ArgumentParser(
        description="Provenance-bound six-passage full-vocabulary retention evaluator"
    )
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser(
        "freeze", help="local-only actual tokenizer assets, no remote code"
    )
    f.add_argument("--model", required=True)
    f.add_argument(
        "--checkpoint",
        required=True,
        help="resolved checkpoint revision, not a host path",
    )
    f.add_argument("--out", required=True)
    s = sub.add_parser(
        "score", help="CPU full-vocabulary logits; float32 then float64 log-softmax"
    )
    s.add_argument("--corpus", required=True)
    s.add_argument("--logits", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--baseline")
    s.add_argument(
        "--changes", help="JSON file declaring exact changed provenance fields"
    )
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--treatment", required=True)
    c.add_argument(
        "--original-diagnostic",
        action="store_true",
        help="report original +0.010-nat diagnostic only",
    )
    c.add_argument("--out")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.action == "freeze":
            frozen = retention.freeze(args.model, args.out, args.checkpoint)
            result = {
                "corpus_sha256": retention.digest(args.out),
                "token_counts": [len(item["input_ids"]) for item in frozen["passages"]],
                "tokenizer_assets_sha256": frozen["tokenizer_assets_sha256"],
            }
        elif args.action == "score":
            result = retention.score(
                args.corpus,
                args.logits,
                args.out,
                args.baseline,
                retention.load(args.changes) if args.changes else None,
            )
        else:
            result = retention.compare(
                args.baseline, args.treatment, args.original_diagnostic
            )
            if args.out:
                retention.save(args.out, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2, "error: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
