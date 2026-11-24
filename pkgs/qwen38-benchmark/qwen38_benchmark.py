#!/usr/bin/env python3
"""Repeatable client-side benchmark for OpenAI-compatible Qwen services.

The harness measures what a client sees: time to first content token, steady
decode rate, delivered tokens per second, and batch aggregate throughput for a
frozen corpus. It never starts a server, touches a Tenstorrent device, or reads
server-private evidence. Point one run at the onix service and another at a
vLLM endpoint; ``compare`` checks that both ran the same workload and that
greedy text stayed identical.

The ``agentic`` mode instead runs concurrent closed-loop agents through a
frozen multi-call trace that reuses its growing history, and reports the
largest concurrency meeting per-call TTFT/ITL and task-time SLOs. It records
client-known prefix overlap separately from any endpoint-reported cached
tokens.

The default corpus is the admitted Qwen3.8-27B B=8 corpus: eight prompts and 64
generated tokens. Greedy decoding, streaming, and a fixed request order keep
the two JSON results comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import socket
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCHEMA = "onix.qwen38-benchmark.v1"
COMPARISON_SCHEMA = "onix.qwen38-benchmark-comparison.v1"
AGENTIC_SCHEMA = "onix.qwen38-agentic-benchmark.v1"
STREAM_DEADLINE_SCHEMA = "onix.qwen38-stream-deadlines.v1"
INTERFERENCE_SCHEMA = "onix.qwen38-prefill-interference.v1"
DEFAULT_MODEL = None
DEFAULT_MAX_TRACE_EVENTS = 4096
DEADLINE_EPS = 1e-12
MAX_RECORDED_MISSED_ORDINALS = 256
DEFAULT_CORPUS = Path(
    os.environ.get("QWEN38_BENCH_CORPUS", Path(__file__).with_name("corpus.json"))
)
DEFAULT_AGENTIC_TRACE = Path(
    os.environ.get(
        "QWEN38_BENCH_AGENTIC_TRACE", Path(__file__).with_name("agentic_trace.json")
    )
)
DEFAULT_INTERFERENCE_WORKLOAD = Path(
    os.environ.get(
        "QWEN38_BENCH_INTERFERENCE_WORKLOAD",
        Path(__file__).with_name("prefill_interference.json"),
    )
)
CACHE_MODES = ("cold", "warm", "disabled", "unknown")
POWER_BOUNDARIES = ("cards", "host", "wall", "unspecified")
CONTROL_MODES = ("short_prompt", "idle")
REQUEST_OUTCOMES = ("completed", "rejected", "timed_out", "incomplete", "unobservable")
EXPLICIT_TOKEN_MAPPING_PROVENANCES = (
    "endpoint_token_ids",
    "endpoint_cumulative_tokens",
    "explicit_timestamps",
)
TIMESTAMP_DOMAIN_SSE = "client_sse_delivery_monotonic_seconds"
TIMESTAMP_DOMAIN_NON_STREAM = "non_stream_wall_only"
GENERATION_TIME_ATTRIBUTION = "client_delivery_only_not_device_generation_time"
OVERFLOW_POLICY = "invalidates_gap_and_deadline_compliance"
SERVER_PHASE_ATTRIBUTION_UNAVAILABLE = {
    "status": "unavailable",
    "reason": "endpoint_only_client_observation_cannot_attribute_server_prefill_vs_decode_phases",
}
ENVIRONMENT_KEYS = (
    "HF_MODEL",
    "MESH_DEVICE",
    "QWEN35_GDN_DECODE_FUSION",
    "QWEN36_GDN_BF4",
    "QWEN36_LM_HEAD_BF4",
    "QWEN36_MLP_DOWN_BF4",
    "QWEN_SDPA_BF8",
    "TT_METAL_DISABLE_SFPLOADMACRO",
    "TT_METAL_HOME",
    "TT_METAL_VERSION",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_TARGET_DEVICE",
    "VLLM_USE_V1",
)


class BenchmarkError(RuntimeError):
    """A target, request, or result cannot satisfy the benchmark contract."""


def validate_header(name: Any, value: Any) -> None:
    if not isinstance(name, str) or not isinstance(value, str):
        raise BenchmarkError("header names and values must be strings")
    if not name or any(character in name + value for character in "\r\n"):
        raise BenchmarkError("header names must be non-empty and contain no newlines")
    try:
        name.encode("latin-1")
        value.encode("latin-1")
    except UnicodeEncodeError as error:
        raise BenchmarkError("headers must be Latin-1 encodable") from error


@dataclass(frozen=True)
class Endpoint:
    label: str
    base_url: str
    model: str | None
    api_key: str | None
    headers: tuple[tuple[str, str], ...]
    timeout: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise BenchmarkError("target label must be a non-empty string")
        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise BenchmarkError(
                f"{self.label}: request timeout must be finite and positive"
            )
        if self.api_key is not None:
            validate_header("Authorization", f"Bearer {self.api_key}")
        if not isinstance(self.headers, tuple):
            raise BenchmarkError(f"{self.label}: headers must be a tuple")
        for pair in self.headers:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise BenchmarkError(f"{self.label}: invalid header pair")
            validate_header(pair[0], pair[1])

    @property
    def origin(self) -> urllib.parse.SplitResult:
        if not isinstance(self.base_url, str):
            raise BenchmarkError(f"{self.label}: base URL must be a string")
        parsed = urllib.parse.urlsplit(self.base_url)
        try:
            hostname = parsed.hostname
            parsed.port
        except ValueError as error:
            raise BenchmarkError(
                f"{self.label}: invalid base URL {self.base_url!r}: {error}"
            ) from error
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise BenchmarkError(f"{self.label}: invalid base URL {self.base_url!r}")
        if parsed.username or parsed.password:
            raise BenchmarkError(
                f"{self.label}: put credentials in --api-key, not the base URL"
            )
        if parsed.query or parsed.fragment:
            raise BenchmarkError(
                f"{self.label}: base URL must not carry a query or fragment"
            )
        return parsed

    def api_path(self, suffix: str) -> str:
        parsed = self.origin
        prefix = parsed.path.rstrip("/")
        if not prefix.endswith("/v1"):
            prefix += "/v1"
        return prefix + suffix

    def root_path(self, suffix: str) -> str:
        parsed = self.origin
        return parsed.path.rstrip("/") + suffix


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def open_connection(
    endpoint: Endpoint, timeout: float | None = None
) -> http.client.HTTPConnection:
    parsed = endpoint.origin
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    effective_timeout = endpoint.timeout if timeout is None else timeout
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            port,
            timeout=effective_timeout,
            context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(
            parsed.hostname, port, timeout=effective_timeout
        )
    connection.connect()
    return connection


def request_headers(endpoint: Endpoint, stream: bool) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream" if stream else "application/json",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    for name, value in endpoint.headers:
        headers[name] = value
    return headers


def read_http_json(
    endpoint: Endpoint,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int, Any]:
    connection = open_connection(endpoint, timeout)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        connection.request(
            method,
            path,
            body=body,
            headers=request_headers(endpoint, False),
        )
        response = connection.getresponse()
        raw = response.read()
        try:
            decoded = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            decoded = raw.decode("utf-8", "replace")[:2048]
        return response.status, decoded
    finally:
        connection.close()


def endpoint_probe(endpoint: Endpoint) -> dict[str, Any]:
    """Best-effort target metadata. It never gates a run."""
    probe: dict[str, Any] = {"base_url": endpoint.base_url}
    probe_timeout = min(endpoint.timeout, 30.0)
    try:
        status, payload = read_http_json(
            endpoint, "GET", endpoint.api_path("/models"), timeout=probe_timeout
        )
        probe["models_status"] = status
        if status == 200 and isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                probe["models"] = [
                    item.get("id")
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
    except (OSError, http.client.HTTPException) as error:
        probe["models_error"] = f"{type(error).__name__}: {error}"

    for suffix in ("/healthz", "/health"):
        try:
            status, payload = read_http_json(
                endpoint, "GET", endpoint.root_path(suffix), timeout=probe_timeout
            )
        except (OSError, http.client.HTTPException):
            continue
        if status == 200:
            probe["health_path"] = suffix
            probe["health"] = payload
            break
    return probe


def resolve_model(endpoint: Endpoint) -> tuple[str, dict[str, Any]]:
    probe = endpoint_probe(endpoint)
    models = probe.get("models")
    if endpoint.model:
        return endpoint.model, probe
    if isinstance(models, list) and models and isinstance(models[0], str):
        return models[0], probe
    raise BenchmarkError(
        f"{endpoint.label}: cannot discover a model from /v1/models; pass --model"
    )


def sse_events(
    response: http.client.HTTPResponse,
    cancel_event: threading.Event | None = None,
):
    """Yield typed SSE frames: ('heartbeat'|'data'|'done'|'cancelled', payload)."""
    data_lines: list[str] = []
    while True:
        if cancel_event is not None and cancel_event.is_set():
            yield ("cancelled", "")
            return
        raw = response.readline()
        if cancel_event is not None and cancel_event.is_set():
            yield ("cancelled", "")
            return
        if not raw:
            if data_lines:
                data = "\n".join(data_lines).strip()
                data_lines.clear()
                if data == "[DONE]":
                    yield ("done", "")
                elif data:
                    yield ("data", data)
            return
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if not data_lines:
                continue
            data = "\n".join(data_lines).strip()
            data_lines.clear()
            if data == "[DONE]":
                yield ("done", "")
                return
            if data:
                yield ("data", data)
            continue
        if line.startswith(":"):
            yield ("heartbeat", line[1:].strip())
            continue
        if not line.startswith("data:"):
            continue
        data_lines.append(line[5:].lstrip())


def sse_payloads(response: http.client.HTTPResponse):
    """Yield one JSON document per SSE event, including multi-line data fields."""
    for kind, payload in sse_events(response):
        if kind == "data":
            yield payload


def content_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _is_role_only_event(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta")
    return isinstance(delta, dict) and isinstance(delta.get("role"), str)


def extract_event_token_mapping(
    payload: dict[str, Any],
) -> tuple[str | None, int | None, str | None]:
    """Extract an explicit endpoint-supplied token-to-event mapping if present.

    Supported explicit sources (never inferred from SSE chunk count or text):
    - endpoint_token_ids: choices[0].token_ids or choices[0].delta.token_ids
      (non-empty list of integer token IDs).
    - endpoint_cumulative_tokens: choices[0].cumulative_completion_tokens or
      usage.completion_tokens emitted on the content event itself (positive int).
    """
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    if isinstance(first, dict):
        raw_token_ids: Any = None
        has_token_ids = False
        if "token_ids" in first:
            raw_token_ids = first["token_ids"]
            has_token_ids = True
        elif isinstance(first.get("delta"), dict) and "token_ids" in first["delta"]:
            raw_token_ids = first["delta"]["token_ids"]
            has_token_ids = True
        if has_token_ids:
            if (
                isinstance(raw_token_ids, list)
                and raw_token_ids
                and all(_is_int(item) for item in raw_token_ids)
            ):
                return ("endpoint_token_ids", len(raw_token_ids), None)
            return ("endpoint_token_ids", None, "invalid_endpoint_token_ids")

        if "cumulative_completion_tokens" in first:
            cumulative = first["cumulative_completion_tokens"]
            if _is_int(cumulative) and cumulative > 0:
                return ("endpoint_cumulative_tokens", cumulative, None)
            return (
                "endpoint_cumulative_tokens",
                None,
                "invalid_endpoint_cumulative_tokens",
            )

    usage = payload.get("usage")
    if isinstance(usage, dict) and "completion_tokens" in usage:
        cumulative = usage["completion_tokens"]
        if _is_int(cumulative) and cumulative > 0:
            return ("endpoint_cumulative_tokens", cumulative, None)
        return (
            "endpoint_cumulative_tokens",
            None,
            "invalid_endpoint_cumulative_tokens",
        )

    return (None, None, None)


def _validate_deadline_slo(
    deadline_slo: dict[str, float | None] | None,
) -> dict[str, float | None]:
    if deadline_slo is None:
        return {
            "ttft_slo_seconds": None,
            "tbt_slo_seconds": None,
            "max_gap_slo_seconds": None,
        }
    if not isinstance(deadline_slo, dict):
        raise BenchmarkError("deadline SLO must be an object")
    allowed = {"ttft_slo_seconds", "tbt_slo_seconds", "max_gap_slo_seconds"}
    unknown = set(deadline_slo) - allowed
    if unknown:
        raise BenchmarkError(f"unknown deadline SLO keys: {sorted(unknown)}")
    normalized: dict[str, float | None] = {}
    for key in ("ttft_slo_seconds", "tbt_slo_seconds", "max_gap_slo_seconds"):
        value = deadline_slo.get(key)
        if value is not None:
            if not _is_finite_number(value) or value <= 0:
                raise BenchmarkError(f"deadline SLO {key} must be finite and positive")
            normalized[key] = float(value)
        else:
            normalized[key] = None
    if (normalized["ttft_slo_seconds"] is None) != (
        normalized["tbt_slo_seconds"] is None
    ):
        raise BenchmarkError(
            "ttft_slo_seconds and tbt_slo_seconds must be supplied together"
        )
    return normalized


def evaluate_stream_timestamps(
    content_event_offsets_seconds: list[float],
    *,
    ttft_slo_seconds: float | None = None,
    tbt_slo_seconds: float | None = None,
    max_gap_slo_seconds: float | None = None,
    token_timestamps_seconds: list[float] | None = None,
    token_mapping_provenance: str = "none",
    token_mapping_reason: str | None = None,
    trace_truncated: bool = False,
    timing_valid: bool = True,
    timing_invalid_reason: str | None = None,
) -> dict[str, Any]:
    """Pure reducer for content-event gaps, max-gap SLOs, and cumulative token deadlines.

    Separates client-observed content-event gaps from per-token cumulative
    deadlines ``TTFT_SLO + (i - 1) * TBT_SLO`` and keeps ``max_gap_within_budget``
    strictly distinct from ``every_token_meets_deadline``.
    """
    slo = _validate_deadline_slo({
        "ttft_slo_seconds": ttft_slo_seconds,
        "tbt_slo_seconds": tbt_slo_seconds,
        "max_gap_slo_seconds": max_gap_slo_seconds,
    })
    ttft_slo = slo["ttft_slo_seconds"]
    tbt_slo = slo["tbt_slo_seconds"]
    max_gap_slo = slo["max_gap_slo_seconds"]

    effective_timing_valid = bool(timing_valid)
    effective_timing_reason = timing_invalid_reason
    if effective_timing_valid:
        prev: float | None = None
        for offset in content_event_offsets_seconds:
            if not _is_finite_number(offset) or offset < 0:
                effective_timing_valid = False
                effective_timing_reason = "non_finite_or_negative_event_timestamp"
                break
            if prev is not None and offset < prev:
                effective_timing_valid = False
                effective_timing_reason = "non_monotonic_event_timestamps"
                break
            prev = float(offset)
    if effective_timing_valid and token_timestamps_seconds is not None:
        prev_tok: float | None = None
        for offset in token_timestamps_seconds:
            if not _is_finite_number(offset) or offset < 0:
                effective_timing_valid = False
                effective_timing_reason = "non_finite_or_negative_token_timestamp"
                break
            if prev_tok is not None and offset < prev_tok:
                effective_timing_valid = False
                effective_timing_reason = "non_monotonic_token_timestamps"
                break
            prev_tok = float(offset)

    ttft_val: float | None = None
    mean_itl_val: float | None = None
    if effective_timing_valid and content_event_offsets_seconds:
        ttft_val = float(content_event_offsets_seconds[0])
        if (
            not trace_truncated
            and len(content_event_offsets_seconds) > 1
            and token_timestamps_seconds is not None
            and token_mapping_provenance in EXPLICIT_TOKEN_MAPPING_PROVENANCES
            and len(token_timestamps_seconds) > 1
        ):
            mean_itl_val = (
                float(token_timestamps_seconds[-1]) - float(token_timestamps_seconds[0])
            ) / (len(token_timestamps_seconds) - 1)
        elif not trace_truncated and len(content_event_offsets_seconds) > 1:
            mean_itl_val = (
                float(content_event_offsets_seconds[-1])
                - float(content_event_offsets_seconds[0])
            ) / (len(content_event_offsets_seconds) - 1)

    if trace_truncated:
        content_event_gaps = {
            "status": "invalid_truncated",
            "gap_count": 0,
            "event_gap_seconds_p50": None,
            "event_gap_seconds_p95": None,
            "event_gap_seconds_p99": None,
            "event_gap_seconds_max": None,
            "event_gap_seconds_mean": None,
        }
    elif not effective_timing_valid:
        content_event_gaps = {
            "status": "invalid_timing",
            "gap_count": 0,
            "event_gap_seconds_p50": None,
            "event_gap_seconds_p95": None,
            "event_gap_seconds_p99": None,
            "event_gap_seconds_max": None,
            "event_gap_seconds_mean": None,
        }
    elif not content_event_offsets_seconds:
        content_event_gaps = {
            "status": "no_content_events",
            "gap_count": 0,
            "event_gap_seconds_p50": None,
            "event_gap_seconds_p95": None,
            "event_gap_seconds_p99": None,
            "event_gap_seconds_max": None,
            "event_gap_seconds_mean": None,
        }
    elif len(content_event_offsets_seconds) == 1:
        content_event_gaps = {
            "status": "single_content_event",
            "gap_count": 0,
            "event_gap_seconds_p50": None,
            "event_gap_seconds_p95": None,
            "event_gap_seconds_p99": None,
            "event_gap_seconds_max": None,
            "event_gap_seconds_mean": None,
        }
    else:
        gaps = [
            float(content_event_offsets_seconds[index])
            - float(content_event_offsets_seconds[index - 1])
            for index in range(1, len(content_event_offsets_seconds))
        ]
        content_event_gaps = {
            "status": "observed",
            "gap_count": len(gaps),
            "event_gap_seconds_p50": percentile(gaps, 0.50),
            "event_gap_seconds_p95": percentile(gaps, 0.95),
            "event_gap_seconds_p99": percentile(gaps, 0.99),
            "event_gap_seconds_max": max(gaps),
            "event_gap_seconds_mean": sum(gaps) / len(gaps),
        }

    # Max-gap compliance (strictly separate from cumulative token deadlines).
    ttft_within: bool | None = None
    if ttft_slo is not None:
        if trace_truncated or not effective_timing_valid or ttft_val is None:
            ttft_within = False
        else:
            ttft_within = ttft_val <= ttft_slo + DEADLINE_EPS

    if max_gap_slo is None:
        max_gap_compliance = {
            "status": "no_slo_configured",
            "max_gap_slo_seconds": None,
            "max_gap_within_budget": None,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": ttft_within,
        }
    elif trace_truncated:
        max_gap_compliance = {
            "status": "invalid_truncated",
            "max_gap_slo_seconds": max_gap_slo,
            "max_gap_within_budget": False,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": False if ttft_slo is not None else None,
        }
    elif not effective_timing_valid:
        max_gap_compliance = {
            "status": "invalid_timing",
            "max_gap_slo_seconds": max_gap_slo,
            "max_gap_within_budget": False,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": False if ttft_slo is not None else None,
        }
    elif content_event_gaps["status"] == "no_content_events":
        max_gap_compliance = {
            "status": "no_content_events",
            "max_gap_slo_seconds": max_gap_slo,
            "max_gap_within_budget": False,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": False if ttft_slo is not None else None,
        }
    elif content_event_gaps["status"] == "single_content_event":
        max_gap_compliance = {
            "status": "single_content_event",
            "max_gap_slo_seconds": max_gap_slo,
            "max_gap_within_budget": True,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": ttft_within,
        }
    else:
        max_gap = content_event_gaps["event_gap_seconds_max"]
        within = max_gap is not None and max_gap <= max_gap_slo + DEADLINE_EPS
        max_gap_compliance = {
            "status": "passed" if within else "failed",
            "max_gap_slo_seconds": max_gap_slo,
            "max_gap_within_budget": within,
            "ttft_slo_seconds": ttft_slo,
            "ttft_within_budget": ttft_within,
        }

    # Cumulative token-ordinal deadline compliance: TTFT_SLO + (i - 1) * TBT_SLO.
    has_explicit_mapping = (
        token_mapping_provenance in EXPLICIT_TOKEN_MAPPING_PROVENANCES
        and token_timestamps_seconds is not None
        and len(token_timestamps_seconds) > 0
    )
    if trace_truncated:
        token_deadline_compliance = {
            "status": "invalid_truncated",
            "reason": "trace_overflow_invalidates_deadline_compliance",
            "provenance": token_mapping_provenance,
            "ttft_slo_seconds": ttft_slo,
            "tbt_slo_seconds": tbt_slo,
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "missed_token_ordinals": [],
            "every_token_meets_deadline": False,
            "worst_deadline_slack_seconds": None,
        }
    elif not effective_timing_valid:
        token_deadline_compliance = {
            "status": "invalid_timing",
            "reason": effective_timing_reason or "invalid_timing_metadata",
            "provenance": token_mapping_provenance,
            "ttft_slo_seconds": ttft_slo,
            "tbt_slo_seconds": tbt_slo,
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "missed_token_ordinals": [],
            "every_token_meets_deadline": False,
            "worst_deadline_slack_seconds": None,
        }
    elif ttft_slo is None or tbt_slo is None:
        token_deadline_compliance = {
            "status": "no_slo_configured",
            "reason": "no_deadline_slo_configured",
            "provenance": token_mapping_provenance if has_explicit_mapping else "none",
            "ttft_slo_seconds": None,
            "tbt_slo_seconds": None,
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "missed_token_ordinals": [],
            "every_token_meets_deadline": None,
            "worst_deadline_slack_seconds": None,
        }
    elif not has_explicit_mapping:
        token_deadline_compliance = {
            "status": "unavailable",
            "reason": token_mapping_reason or "no_explicit_token_to_event_mapping",
            "provenance": "none",
            "ttft_slo_seconds": ttft_slo,
            "tbt_slo_seconds": tbt_slo,
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "missed_token_ordinals": [],
            "every_token_meets_deadline": None,
            "worst_deadline_slack_seconds": None,
        }
    else:
        assert token_timestamps_seconds is not None
        missed_ordinals: list[int] = []
        missed_count = 0
        worst_slack: float | None = None
        for ordinal, arrival in enumerate(token_timestamps_seconds, start=1):
            deadline = ttft_slo + (ordinal - 1) * tbt_slo
            slack = deadline - float(arrival)
            if worst_slack is None or slack < worst_slack:
                worst_slack = slack
            if float(arrival) > deadline + DEADLINE_EPS:
                missed_count += 1
                if len(missed_ordinals) < MAX_RECORDED_MISSED_ORDINALS:
                    missed_ordinals.append(ordinal)
        evaluated_tokens = len(token_timestamps_seconds)
        met_tokens = evaluated_tokens - missed_count
        every_ok = missed_count == 0
        token_deadline_compliance = {
            "status": "passed" if every_ok else "failed",
            "reason": None if every_ok else "token_deadline_missed",
            "provenance": token_mapping_provenance,
            "ttft_slo_seconds": ttft_slo,
            "tbt_slo_seconds": tbt_slo,
            "evaluated_tokens": evaluated_tokens,
            "met_tokens": met_tokens,
            "missed_tokens": missed_count,
            "missed_token_ordinals": missed_ordinals,
            "every_token_meets_deadline": every_ok,
            "worst_deadline_slack_seconds": worst_slack,
        }

    return {
        "timestamp_domain": TIMESTAMP_DOMAIN_SSE,
        "generation_time_attribution": GENERATION_TIME_ATTRIBUTION,
        "timing_valid": effective_timing_valid,
        "timing_invalid_reason": effective_timing_reason,
        "trace_truncated": bool(trace_truncated),
        "ttft_seconds": ttft_val,
        "mean_itl_seconds": mean_itl_val,
        "content_event_gaps": content_event_gaps,
        "max_gap_compliance": max_gap_compliance,
        "token_deadline_compliance": token_deadline_compliance,
    }


def _build_non_stream_observation(
    deadline_slo: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    slo = _validate_deadline_slo(deadline_slo)
    return {
        "schema": STREAM_DEADLINE_SCHEMA,
        "timestamp_domain": TIMESTAMP_DOMAIN_NON_STREAM,
        "generation_time_attribution": GENERATION_TIME_ATTRIBUTION,
        "overflow_policy": OVERFLOW_POLICY,
        "max_trace_events": 0,
        "recorded_trace_events": 0,
        "dropped_trace_events": 0,
        "trace_truncated": False,
        "overflow": False,
        "timing_valid": True,
        "timing_invalid_reason": None,
        "done_seen": False,
        "finish_reason_seen": False,
        "finish_reason": None,
        "event_counts": {
            "content": 0,
            "heartbeat": 0,
            "role_only": 0,
            "usage_only": 0,
            "empty": 0,
            "done": 0,
        },
        "content_event_offsets_seconds": [],
        "content_event_gaps": {
            "status": "unavailable_non_stream",
            "gap_count": 0,
            "event_gap_seconds_p50": None,
            "event_gap_seconds_p95": None,
            "event_gap_seconds_p99": None,
            "event_gap_seconds_max": None,
            "event_gap_seconds_mean": None,
        },
        "token_mapping": {
            "status": "unavailable",
            "provenance": "none",
            "reason": "non_stream_request",
            "mapped_tokens": 0,
        },
        "token_deadline_compliance": {
            "status": "unavailable",
            "reason": "non_stream_request",
            "provenance": "none",
            "ttft_slo_seconds": slo["ttft_slo_seconds"],
            "tbt_slo_seconds": slo["tbt_slo_seconds"],
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "missed_token_ordinals": [],
            "every_token_meets_deadline": None,
            "worst_deadline_slack_seconds": None,
        },
        "max_gap_compliance": {
            "status": "unavailable_non_stream",
            "max_gap_slo_seconds": slo["max_gap_slo_seconds"],
            "max_gap_within_budget": None,
            "ttft_slo_seconds": slo["ttft_slo_seconds"],
            "ttft_within_budget": None,
        },
    }


def record_from_usage(
    text: str,
    usage: dict[str, Any] | None,
    chunk_tokens: int,
    start: float,
    end: float,
    first_token: float | None,
    last_token: float | None,
    *,
    text_sha256: str | None = None,
    text_characters: int | None = None,
) -> dict[str, Any]:
    completion_tokens = None
    prompt_tokens = None
    token_count_source = "chunks"
    if isinstance(usage, dict):
        raw_completion = usage.get("completion_tokens")
        raw_prompt = usage.get("prompt_tokens")
        if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
            completion_tokens = raw_completion
            token_count_source = "usage"
        if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool):
            prompt_tokens = raw_prompt
    if completion_tokens is None:
        completion_tokens = chunk_tokens

    ttft = (
        None
        if first_token is None
        or not _is_finite_number(first_token)
        or not _is_finite_number(start)
        else first_token - start
    )
    last = (
        None
        if last_token is None
        or not _is_finite_number(last_token)
        or not _is_finite_number(start)
        else last_token - start
    )
    wall = (
        None
        if not _is_finite_number(end) or not _is_finite_number(start)
        else end - start
    )
    decode_rate = None
    if (
        ttft is not None
        and last is not None
        and last > ttft
        and completion_tokens is not None
        and completion_tokens > 1
    ):
        decode_rate = (completion_tokens - 1) / (last - ttft)
    e2e_rate = (
        completion_tokens / wall
        if completion_tokens is not None
        and completion_tokens > 0
        and wall is not None
        and wall > 0
        else None
    )
    return {
        "text_sha256": (
            text_sha256
            if text_sha256 is not None
            else hashlib.sha256(text.encode()).hexdigest()
        ),
        "text_characters": len(text) if text_characters is None else text_characters,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_count_source": token_count_source,
        "stream_chunks": chunk_tokens,
        "ttft_seconds": ttft,
        "last_token_seconds": last,
        "wall_seconds": wall,
        "decode_tokens_per_second": decode_rate,
        "e2e_tokens_per_second": e2e_rate,
    }


def _resolve_token_timestamps(
    event_offsets: list[float],
    event_mappings: list[tuple[str | None, int | None, str | None]],
    completion_tokens: int | None,
    *,
    trace_truncated: bool,
    timing_valid: bool,
) -> tuple[list[float] | None, str, str | None, dict[str, Any]]:
    if trace_truncated:
        return (
            None,
            "none",
            "trace_overflow_invalidates_token_mapping",
            {
                "status": "invalid_truncated",
                "provenance": "none",
                "reason": "trace_overflow_invalidates_token_mapping",
                "mapped_tokens": 0,
            },
        )
    if not timing_valid:
        return (
            None,
            "none",
            "invalid_timing_metadata",
            {
                "status": "invalid_timing",
                "provenance": "none",
                "reason": "invalid_timing_metadata",
                "mapped_tokens": 0,
            },
        )
    if not event_offsets or not _is_int(completion_tokens) or completion_tokens <= 0:
        return (
            None,
            "none",
            "no_explicit_token_to_event_mapping",
            {
                "status": "unavailable",
                "provenance": "none",
                "reason": "no_explicit_token_to_event_mapping",
                "mapped_tokens": 0,
            },
        )
    kinds = {kind for kind, _val, _err in event_mappings}
    errors = [err for _kind, _val, err in event_mappings if err is not None]
    if kinds == {None}:
        return (
            None,
            "none",
            "no_explicit_token_to_event_mapping",
            {
                "status": "unavailable",
                "provenance": "none",
                "reason": "no_explicit_token_to_event_mapping",
                "mapped_tokens": 0,
            },
        )
    if errors or None in kinds or len(kinds) != 1:
        reason = errors[0] if errors else "incomplete_or_mixed_token_mapping"
        return (
            None,
            "none",
            reason,
            {
                "status": "unavailable",
                "provenance": "none",
                "reason": reason,
                "mapped_tokens": 0,
            },
        )
    provenance = next(iter(kinds))
    assert provenance is not None
    token_timestamps: list[float] = []
    if provenance == "endpoint_token_ids":
        for offset, (_kind, count, _err) in zip(
            event_offsets, event_mappings, strict=True
        ):
            assert count is not None
            token_timestamps.extend([offset] * count)
        if len(token_timestamps) != completion_tokens:
            return (
                None,
                "none",
                "invalid_token_mapping_count_mismatch",
                {
                    "status": "unavailable",
                    "provenance": "none",
                    "reason": "invalid_token_mapping_count_mismatch",
                    "mapped_tokens": 0,
                },
            )
    elif provenance == "endpoint_cumulative_tokens":
        previous_cumulative = 0
        for offset, (_kind, cumulative, _err) in zip(
            event_offsets, event_mappings, strict=True
        ):
            assert cumulative is not None
            if cumulative <= previous_cumulative:
                return (
                    None,
                    "none",
                    "invalid_cumulative_token_mapping",
                    {
                        "status": "unavailable",
                        "provenance": "none",
                        "reason": "invalid_cumulative_token_mapping",
                        "mapped_tokens": 0,
                    },
                )
            token_timestamps.extend([offset] * (cumulative - previous_cumulative))
            previous_cumulative = cumulative
        if previous_cumulative != completion_tokens:
            return (
                None,
                "none",
                "invalid_token_mapping_count_mismatch",
                {
                    "status": "unavailable",
                    "provenance": "none",
                    "reason": "invalid_token_mapping_count_mismatch",
                    "mapped_tokens": 0,
                },
            )
    else:
        return (
            None,
            "none",
            "unsupported_token_mapping_provenance",
            {
                "status": "unavailable",
                "provenance": "none",
                "reason": "unsupported_token_mapping_provenance",
                "mapped_tokens": 0,
            },
        )
    return (
        token_timestamps,
        provenance,
        None,
        {
            "status": "observed",
            "provenance": provenance,
            "reason": None,
            "mapped_tokens": len(token_timestamps),
        },
    )


def _failure_stream_record(
    *,
    outcome: str,
    outcome_reason: str,
    error_message: str,
    start: float,
    end: float,
    max_trace_events: int,
    deadline_slo: dict[str, float | None] | None,
    event_counts: dict[str, int] | None = None,
    content_offsets: list[float] | None = None,
    dropped_trace_events: int = 0,
    trace_truncated: bool = False,
    done_seen: bool = False,
    finish_reason_seen: bool = False,
    finish_reason: str | None = None,
    text_sha256: str | None = None,
    text_characters: int = 0,
    chunk_tokens: int = 0,
    usage: dict[str, Any] | None = None,
    first_token: float | None = None,
    last_token: float | None = None,
    timing_valid: bool = True,
    timing_invalid_reason: str | None = None,
) -> dict[str, Any]:
    slo = _validate_deadline_slo(deadline_slo)
    wall: float | None = None
    if _is_finite_number(start) and _is_finite_number(end) and end >= start:
        wall = end - start
    else:
        timing_valid = False
        timing_invalid_reason = timing_invalid_reason or "invalid_wall_timestamps"
    safe_offsets = list(content_offsets or [])
    if not timing_valid:
        safe_offsets = [
            float(val)
            for val in safe_offsets
            if _is_finite_number(val) and val >= 0
        ]
    reduced = evaluate_stream_timestamps(
        safe_offsets,
        ttft_slo_seconds=slo["ttft_slo_seconds"],
        tbt_slo_seconds=slo["tbt_slo_seconds"],
        max_gap_slo_seconds=slo["max_gap_slo_seconds"],
        trace_truncated=trace_truncated,
        timing_valid=timing_valid,
        timing_invalid_reason=timing_invalid_reason,
    )
    ttft_val = (
        first_token - start
        if timing_valid
        and first_token is not None
        and _is_finite_number(first_token)
        and _is_finite_number(start)
        and first_token >= start
        else None
    )
    last_val = (
        last_token - start
        if timing_valid
        and last_token is not None
        and _is_finite_number(last_token)
        and _is_finite_number(start)
        and last_token >= start
        else None
    )
    prompt_tokens = None
    completion_tokens = None
    token_count_source = "unavailable"
    if isinstance(usage, dict):
        if _is_int(usage.get("prompt_tokens")):
            prompt_tokens = usage["prompt_tokens"]
        if _is_int(usage.get("completion_tokens")):
            completion_tokens = usage["completion_tokens"]
            token_count_source = "usage"
    counts = event_counts or {
        "content": chunk_tokens,
        "heartbeat": 0,
        "role_only": 0,
        "usage_only": 0,
        "empty": 0,
        "done": 1 if done_seen else 0,
    }
    return {
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "error": error_message,
        "text_sha256": text_sha256,
        "text_characters": text_characters,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_count_source": token_count_source,
        "stream_chunks": chunk_tokens,
        "ttft_seconds": ttft_val,
        "last_token_seconds": last_val,
        "wall_seconds": wall,
        "decode_tokens_per_second": None,
        "e2e_tokens_per_second": None,
        "finish_reason_seen": finish_reason_seen,
        "stream_observation": {
            "schema": STREAM_DEADLINE_SCHEMA,
            "timestamp_domain": TIMESTAMP_DOMAIN_SSE,
            "generation_time_attribution": GENERATION_TIME_ATTRIBUTION,
            "overflow_policy": OVERFLOW_POLICY,
            "max_trace_events": max_trace_events,
            "recorded_trace_events": len(safe_offsets),
            "dropped_trace_events": dropped_trace_events,
            "trace_truncated": trace_truncated,
            "overflow": trace_truncated,
            "timing_valid": reduced["timing_valid"],
            "timing_invalid_reason": reduced["timing_invalid_reason"],
            "done_seen": done_seen,
            "finish_reason_seen": finish_reason_seen,
            "finish_reason": finish_reason,
            "event_counts": counts,
            "content_event_offsets_seconds": safe_offsets,
            "content_event_gaps": reduced["content_event_gaps"],
            "token_mapping": {
                "status": "invalid_truncated"
                if trace_truncated
                else ("invalid_timing" if not reduced["timing_valid"] else "unavailable"),
                "provenance": "none",
                "reason": outcome_reason,
                "mapped_tokens": 0,
            },
            "token_deadline_compliance": reduced["token_deadline_compliance"],
            "max_gap_compliance": reduced["max_gap_compliance"],
        },
    }


def observe_stream_completion(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    path: str,
    payload: dict[str, Any],
    context: str,
    start: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Stream one completion with bounded event tracking and non-collapsing outcomes."""
    if not _is_int(max_trace_events) or max_trace_events <= 0:
        raise BenchmarkError("max_trace_events must be a positive integer")
    slo = _validate_deadline_slo(deadline_slo)

    first_token: float | None = None
    last_token: float | None = None
    text_hasher = hashlib.sha256()
    text_characters = 0
    chunk_tokens = 0
    usage: dict[str, Any] | None = None
    finish_reason_seen = False
    finish_reason: str | None = None
    done_seen = False
    trace_truncated = False
    dropped_trace_events = 0
    content_offsets: list[float] = []
    event_mappings: list[tuple[str | None, int | None, str | None]] = []
    event_counts = {
        "content": 0,
        "heartbeat": 0,
        "role_only": 0,
        "usage_only": 0,
        "empty": 0,
        "done": 0,
    }
    timing_valid = _is_finite_number(start)
    timing_invalid_reason: str | None = (
        None if timing_valid else "non_finite_start_timestamp"
    )
    previous_offset: float | None = None
    response: http.client.HTTPResponse | None = None

    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers=request_headers(endpoint, True),
        )
        response = connection.getresponse()
        if response.status != 200:
            body = response.read().decode("utf-8", "replace")
            end = clock()
            return (
                _failure_stream_record(
                    outcome="rejected",
                    outcome_reason=f"http_{response.status}",
                    error_message=f"{endpoint.label}: HTTP {response.status}: {body[:1000]}",
                    start=start,
                    end=end,
                    max_trace_events=max_trace_events,
                    deadline_slo=slo,
                    event_counts=event_counts,
                ),
                None,
            )

        for frame_kind, frame_payload in sse_events(response, cancel_event=cancel_event):
            if frame_kind == "cancelled":
                end = clock()
                return (
                    _failure_stream_record(
                        outcome="incomplete",
                        outcome_reason="cancelled",
                        error_message=f"{endpoint.label}: stream cancelled for {context}",
                        start=start,
                        end=end,
                        max_trace_events=max_trace_events,
                        deadline_slo=slo,
                        event_counts=event_counts,
                        content_offsets=content_offsets,
                        dropped_trace_events=dropped_trace_events,
                        trace_truncated=trace_truncated,
                        done_seen=done_seen,
                        finish_reason_seen=finish_reason_seen,
                        finish_reason=finish_reason,
                        text_sha256=(
                            text_hasher.hexdigest() if text_characters > 0 else None
                        ),
                        text_characters=text_characters,
                        chunk_tokens=chunk_tokens,
                        usage=usage,
                        first_token=first_token,
                        last_token=last_token,
                        timing_valid=timing_valid,
                        timing_invalid_reason=timing_invalid_reason,
                    ),
                    usage,
                )
            if frame_kind == "heartbeat":
                event_counts["heartbeat"] += 1
                continue
            if frame_kind == "done":
                done_seen = True
                event_counts["done"] += 1
                break
            try:
                decoded = json.loads(frame_payload)
            except json.JSONDecodeError as error:
                end = clock()
                return (
                    _failure_stream_record(
                        outcome="incomplete",
                        outcome_reason="invalid_sse_json",
                        error_message=(
                            f"{endpoint.label}: invalid SSE payload "
                            f"{frame_payload[:300]!r}: {error}"
                        ),
                        start=start,
                        end=end,
                        max_trace_events=max_trace_events,
                        deadline_slo=slo,
                        event_counts=event_counts,
                        content_offsets=content_offsets,
                        dropped_trace_events=dropped_trace_events,
                        trace_truncated=trace_truncated,
                        done_seen=done_seen,
                        finish_reason_seen=finish_reason_seen,
                        finish_reason=finish_reason,
                        text_sha256=(
                            text_hasher.hexdigest() if text_characters > 0 else None
                        ),
                        text_characters=text_characters,
                        chunk_tokens=chunk_tokens,
                        usage=usage,
                        first_token=first_token,
                        last_token=last_token,
                        timing_valid=timing_valid,
                        timing_invalid_reason=timing_invalid_reason,
                    ),
                    usage,
                )
            if not isinstance(decoded, dict):
                event_counts["empty"] += 1
                continue
            if isinstance(decoded.get("error"), dict):
                end = clock()
                return (
                    _failure_stream_record(
                        outcome="rejected",
                        outcome_reason="endpoint_error_payload",
                        error_message=f"{endpoint.label}: {decoded['error']}",
                        start=start,
                        end=end,
                        max_trace_events=max_trace_events,
                        deadline_slo=slo,
                        event_counts=event_counts,
                        content_offsets=content_offsets,
                        dropped_trace_events=dropped_trace_events,
                        trace_truncated=trace_truncated,
                        done_seen=done_seen,
                        finish_reason_seen=finish_reason_seen,
                        finish_reason=finish_reason,
                        text_sha256=(
                            text_hasher.hexdigest() if text_characters > 0 else None
                        ),
                        text_characters=text_characters,
                        chunk_tokens=chunk_tokens,
                        usage=usage,
                        first_token=first_token,
                        last_token=last_token,
                        timing_valid=timing_valid,
                        timing_invalid_reason=timing_invalid_reason,
                    ),
                    usage,
                )
            has_usage = isinstance(decoded.get("usage"), dict)
            if has_usage:
                usage = decoded["usage"]
            text = content_text(decoded)
            if text:
                now = clock()
                if first_token is None:
                    first_token = now
                last_token = now
                text_hasher.update(text.encode())
                text_characters += len(text)
                chunk_tokens += 1
                event_counts["content"] += 1
                if timing_valid:
                    if not _is_finite_number(now):
                        timing_valid = False
                        timing_invalid_reason = "non_finite_event_timestamp"
                    else:
                        offset = float(now) - float(start)
                        if not _is_finite_number(offset) or offset < 0:
                            timing_valid = False
                            timing_invalid_reason = "negative_event_offset"
                        elif previous_offset is not None and offset < previous_offset:
                            timing_valid = False
                            timing_invalid_reason = "non_monotonic_event_timestamps"
                        else:
                            previous_offset = offset
                            if len(content_offsets) < max_trace_events:
                                content_offsets.append(offset)
                                event_mappings.append(
                                    extract_event_token_mapping(decoded)
                                )
                            else:
                                trace_truncated = True
                                dropped_trace_events += 1
                else:
                    if len(content_offsets) >= max_trace_events:
                        trace_truncated = True
                        dropped_trace_events += 1
            elif _is_role_only_event(decoded):
                event_counts["role_only"] += 1
            elif has_usage:
                event_counts["usage_only"] += 1
            else:
                event_counts["empty"] += 1
            choices = decoded.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict) and first.get("finish_reason") is not None:
                    finish_reason_seen = True
                    finish_reason = str(first["finish_reason"])
        end = clock()
    except (TimeoutError, socket.timeout) as error:
        end = clock()
        return (
            _failure_stream_record(
                outcome="timed_out",
                outcome_reason="request_timeout",
                error_message=f"{endpoint.label}: timeout for {context}: {error}",
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=(text_hasher.hexdigest() if text_characters > 0 else None),
                text_characters=text_characters,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=timing_valid,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )
    except (OSError, http.client.HTTPException) as error:
        end = clock()
        return (
            _failure_stream_record(
                outcome="incomplete",
                outcome_reason=f"transport_{type(error).__name__}",
                error_message=(
                    f"{endpoint.label}: transport error for {context}: "
                    f"{type(error).__name__}: {error}"
                ),
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=(text_hasher.hexdigest() if text_characters > 0 else None),
                text_characters=text_characters,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=timing_valid,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )
    finally:
        if response is not None:
            response.close()

    if timing_valid:
        if not _is_finite_number(end) or end < start:
            timing_valid = False
            timing_invalid_reason = "invalid_end_timestamp"
        elif last_token is not None and (
            not _is_finite_number(last_token) or last_token > end
        ):
            timing_valid = False
            timing_invalid_reason = "event_timestamp_exceeds_wall"
        elif end <= start and chunk_tokens > 0:
            timing_valid = False
            timing_invalid_reason = "non_positive_wall_seconds"

    if first_token is None or text_characters == 0:
        return (
            _failure_stream_record(
                outcome="incomplete",
                outcome_reason="empty_stream_content",
                error_message=f"{endpoint.label}: stream produced no content for {context}",
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=None,
                text_characters=0,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=timing_valid,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )

    if not finish_reason_seen or not done_seen:
        return (
            _failure_stream_record(
                outcome="incomplete",
                outcome_reason="eof_without_valid_completion",
                error_message=(
                    f"{endpoint.label}: stream reached EOF without valid completion "
                    f"(finish_reason_seen={finish_reason_seen}, done_seen={done_seen}) "
                    f"for {context}"
                ),
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=text_hasher.hexdigest(),
                text_characters=text_characters,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=timing_valid,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )

    if (
        not isinstance(usage, dict)
        or not _is_int(usage.get("completion_tokens"))
        or usage["completion_tokens"] <= 0
    ):
        return (
            _failure_stream_record(
                outcome="incomplete",
                outcome_reason="missing_usage_completion_tokens",
                error_message=(
                    f"{endpoint.label}: stream did not report a positive "
                    f"usage.completion_tokens for {context}"
                ),
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=text_hasher.hexdigest(),
                text_characters=text_characters,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=timing_valid,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )

    if not timing_valid:
        return (
            _failure_stream_record(
                outcome="unobservable",
                outcome_reason=timing_invalid_reason or "invalid_timing_metadata",
                error_message=(
                    f"{endpoint.label}: invalid timing metadata "
                    f"({timing_invalid_reason}) for {context}"
                ),
                start=start,
                end=end,
                max_trace_events=max_trace_events,
                deadline_slo=slo,
                event_counts=event_counts,
                content_offsets=content_offsets,
                dropped_trace_events=dropped_trace_events,
                trace_truncated=trace_truncated,
                done_seen=done_seen,
                finish_reason_seen=finish_reason_seen,
                finish_reason=finish_reason,
                text_sha256=text_hasher.hexdigest(),
                text_characters=text_characters,
                chunk_tokens=chunk_tokens,
                usage=usage,
                first_token=first_token,
                last_token=last_token,
                timing_valid=False,
                timing_invalid_reason=timing_invalid_reason,
            ),
            usage,
        )

    token_timestamps, token_provenance, token_reason, token_mapping = (
        _resolve_token_timestamps(
            content_offsets,
            event_mappings,
            usage["completion_tokens"],
            trace_truncated=trace_truncated,
            timing_valid=timing_valid,
        )
    )
    reduced = evaluate_stream_timestamps(
        content_offsets,
        ttft_slo_seconds=slo["ttft_slo_seconds"],
        tbt_slo_seconds=slo["tbt_slo_seconds"],
        max_gap_slo_seconds=slo["max_gap_slo_seconds"],
        token_timestamps_seconds=token_timestamps,
        token_mapping_provenance=token_provenance,
        token_mapping_reason=token_reason,
        trace_truncated=trace_truncated,
        timing_valid=timing_valid,
        timing_invalid_reason=timing_invalid_reason,
    )
    record = record_from_usage(
        "",
        usage,
        chunk_tokens,
        start,
        end,
        first_token,
        last_token,
        text_sha256=text_hasher.hexdigest(),
        text_characters=text_characters,
    )
    record["finish_reason_seen"] = finish_reason_seen
    record["outcome"] = "unobservable" if trace_truncated else "completed"
    record["outcome_reason"] = "trace_overflow" if trace_truncated else "ok"
    record["error"] = (
        f"{endpoint.label}: trace overflow ({dropped_trace_events} dropped events) for {context}"
        if trace_truncated
        else None
    )
    record["stream_observation"] = {
        "schema": STREAM_DEADLINE_SCHEMA,
        "timestamp_domain": TIMESTAMP_DOMAIN_SSE,
        "generation_time_attribution": GENERATION_TIME_ATTRIBUTION,
        "overflow_policy": OVERFLOW_POLICY,
        "max_trace_events": max_trace_events,
        "recorded_trace_events": len(content_offsets),
        "dropped_trace_events": dropped_trace_events,
        "trace_truncated": trace_truncated,
        "overflow": trace_truncated,
        "timing_valid": reduced["timing_valid"],
        "timing_invalid_reason": reduced["timing_invalid_reason"],
        "done_seen": done_seen,
        "finish_reason_seen": finish_reason_seen,
        "finish_reason": finish_reason,
        "event_counts": event_counts,
        "content_event_offsets_seconds": content_offsets,
        "content_event_gaps": reduced["content_event_gaps"],
        "token_mapping": token_mapping,
        "token_deadline_compliance": reduced["token_deadline_compliance"],
        "max_gap_compliance": reduced["max_gap_compliance"],
    }
    return record, usage


def stream_completion(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    path: str,
    payload: dict[str, Any],
    context: str,
    start: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream one completion and measure it. Raises BenchmarkError on failed stream."""
    record, usage = observe_stream_completion(
        connection,
        endpoint,
        path,
        payload,
        context,
        start,
        clock=clock,
        max_trace_events=max_trace_events,
        deadline_slo=deadline_slo,
        cancel_event=cancel_event,
    )
    if record["outcome"] not in ("completed", "unobservable") or usage is None:
        raise BenchmarkError(record.get("error") or f"{endpoint.label}: stream failed")
    return record, usage


def perform_request(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    start: float | None = None,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    request_start = clock() if start is None else start
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    if stream:
        record, _usage = stream_completion(
            connection,
            endpoint,
            endpoint.api_path("/completions"),
            payload,
            f"prompt {prompt[:60]!r}",
            request_start,
            clock=clock,
            max_trace_events=max_trace_events,
            deadline_slo=deadline_slo,
        )
        return record

    connection.request(
        "POST",
        endpoint.api_path("/completions"),
        body=json.dumps(payload).encode(),
        headers=request_headers(endpoint, False),
    )
    response = connection.getresponse()
    if response.status != 200:
        body = response.read().decode("utf-8", "replace")
        raise BenchmarkError(f"{endpoint.label}: HTTP {response.status}: {body[:1000]}")

    raw = response.read()
    end = clock()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            f"{endpoint.label}: non-JSON response: {raw[:300]!r}"
        ) from error
    if not isinstance(decoded, dict):
        raise BenchmarkError(f"{endpoint.label}: response is not an object")
    if isinstance(decoded.get("error"), dict):
        raise BenchmarkError(f"{endpoint.label}: {decoded['error']}")
    usage = decoded.get("usage")
    if not isinstance(usage, dict) or not _is_int(usage.get("completion_tokens")):
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response did not report completion_tokens"
        )
    text = content_text(decoded)
    if not text:
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response had no completion text"
        )
    record = record_from_usage(
        text,
        usage,
        1,
        request_start,
        end,
        None,
        None,
    )
    if record["completion_tokens"] <= 0:
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response reported no completion tokens"
        )
    record["outcome"] = "completed"
    record["outcome_reason"] = "ok"
    record["error"] = None
    record["stream_observation"] = _build_non_stream_observation(deadline_slo)
    return record


def summarize_population_and_deadlines(
    records: list[dict[str, Any]],
    *,
    duration_seconds: float | None,
    arrival_mode: str,
    offered_load: dict[str, Any],
    warmup_runs: int,
    deadline_slo: dict[str, float | None] | None = None,
    legacy_slo_met: bool | None = None,
) -> dict[str, Any]:
    """Summarize full request population outcomes and versioned stream-deadline metrics.

    Every offered request remains in the denominator; no failed, truncated,
    cancelled, or invalid-timing request is ever dropped to become compliant.
    """
    slo = _validate_deadline_slo(deadline_slo)
    max_gap_slo = slo["max_gap_slo_seconds"]
    ttft_slo = slo["ttft_slo_seconds"]
    tbt_slo = slo["tbt_slo_seconds"]

    offered = len(records)
    outcome_counts = {outcome: 0 for outcome in REQUEST_OUTCOMES}
    cancelled_count = 0
    reason_counts: dict[str, int] = {}

    all_event_gaps: list[float] = []
    gap_request_count = 0
    single_event_request_count = 0
    per_request_max_gaps: list[float] = []
    per_request_p95_gaps: list[float] = []

    max_gap_compliant = 0
    max_gap_violating = 0
    max_gap_unobservable_or_failed = 0

    token_evaluated_requests = 0
    token_compliant_requests = 0
    token_violating_requests = 0
    token_unavailable_requests = 0
    token_failed_or_invalid_requests = 0
    token_provenances: set[str] = set()
    total_evaluated_tokens = 0
    total_met_tokens = 0
    total_missed_tokens = 0

    any_sse = False
    any_non_stream = False

    for record in records:
        outcome = record.get("outcome", "completed")
        if outcome not in outcome_counts:
            outcome = "unobservable"
        outcome_counts[outcome] += 1
        reason = str(record.get("outcome_reason") or "ok")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason == "cancelled":
            cancelled_count += 1

        obs = record.get("stream_observation")
        if isinstance(obs, dict):
            domain = obs.get("timestamp_domain")
            if domain == TIMESTAMP_DOMAIN_SSE:
                any_sse = True
            elif domain == TIMESTAMP_DOMAIN_NON_STREAM:
                any_non_stream = True
            tok_map = obs.get("token_mapping")
            if isinstance(tok_map, dict) and isinstance(tok_map.get("provenance"), str):
                token_provenances.add(tok_map["provenance"])
            else:
                token_provenances.add("none")
        else:
            token_provenances.add("none")

        if outcome == "completed" and isinstance(obs, dict):
            gap_block = obs.get("content_event_gaps") or {}
            gap_status = gap_block.get("status")
            if gap_status == "observed":
                offsets = obs.get("content_event_offsets_seconds") or []
                req_gaps = [
                    float(offsets[index]) - float(offsets[index - 1])
                    for index in range(1, len(offsets))
                ]
                if req_gaps:
                    all_event_gaps.extend(req_gaps)
                    gap_request_count += 1
                    per_request_max_gaps.append(max(req_gaps))
                    req_p95 = percentile(req_gaps, 0.95)
                    if req_p95 is not None:
                        per_request_p95_gaps.append(req_p95)
            elif gap_status == "single_content_event":
                single_event_request_count += 1

            # Max-gap compliance accounting (strictly separate from cumulative deadlines).
            if max_gap_slo is not None:
                mgc = obs.get("max_gap_compliance") or {}
                within = mgc.get("max_gap_within_budget")
                if within is True:
                    max_gap_compliant += 1
                elif within is False:
                    max_gap_violating += 1
                else:
                    max_gap_unobservable_or_failed += 1

            # Cumulative token-deadline compliance accounting.
            if ttft_slo is not None and tbt_slo is not None:
                tdc = obs.get("token_deadline_compliance") or {}
                tdc_status = tdc.get("status")
                if tdc_status in ("passed", "failed"):
                    token_evaluated_requests += 1
                    if tdc.get("every_token_meets_deadline") is True:
                        token_compliant_requests += 1
                    else:
                        token_violating_requests += 1
                    ev_toks = tdc.get("evaluated_tokens")
                    met_toks = tdc.get("met_tokens")
                    miss_toks = tdc.get("missed_tokens")
                    if _is_int(ev_toks):
                        total_evaluated_tokens += ev_toks
                    if _is_int(met_toks):
                        total_met_tokens += met_toks
                    if _is_int(miss_toks):
                        total_missed_tokens += miss_toks
                elif tdc_status == "unavailable":
                    token_unavailable_requests += 1
                else:
                    token_failed_or_invalid_requests += 1
        else:
            if max_gap_slo is not None:
                max_gap_unobservable_or_failed += 1
            if ttft_slo is not None and tbt_slo is not None:
                token_failed_or_invalid_requests += 1

    completed = outcome_counts["completed"]
    rejected = outcome_counts["rejected"]
    timed_out = outcome_counts["timed_out"]
    incomplete = outcome_counts["incomplete"]
    unobservable = outcome_counts["unobservable"]
    excluded_gap_requests = offered - (gap_request_count + single_event_request_count)

    # Max-gap SLO rollup.
    if max_gap_slo is None:
        max_gap_slo_compliance = {
            "status": "no_slo_configured",
            "max_gap_slo_seconds": None,
            "denominator_requests": offered,
            "compliant_requests": 0,
            "violating_requests": 0,
            "unobservable_or_failed_requests": offered - completed,
            "compliance_rate": None,
            "every_request_meets_max_gap_slo": None,
        }
    elif offered == 0:
        max_gap_slo_compliance = {
            "status": "empty_population",
            "max_gap_slo_seconds": max_gap_slo,
            "denominator_requests": 0,
            "compliant_requests": 0,
            "violating_requests": 0,
            "unobservable_or_failed_requests": 0,
            "compliance_rate": None,
            "every_request_meets_max_gap_slo": False,
        }
    else:
        every_max_gap = (
            max_gap_compliant == offered
            and max_gap_violating == 0
            and max_gap_unobservable_or_failed == 0
        )
        mg_status = (
            "fail_closed_incomplete_population"
            if max_gap_unobservable_or_failed > 0
            else ("passed" if every_max_gap else "failed")
        )
        max_gap_slo_compliance = {
            "status": mg_status,
            "max_gap_slo_seconds": max_gap_slo,
            "denominator_requests": offered,
            "compliant_requests": max_gap_compliant,
            "violating_requests": max_gap_violating,
            "unobservable_or_failed_requests": max_gap_unobservable_or_failed,
            "compliance_rate": max_gap_compliant / offered,
            "every_request_meets_max_gap_slo": every_max_gap,
        }

    # Cumulative token-deadline SLO rollup.
    if ttft_slo is None or tbt_slo is None:
        token_status = "no_slo_configured"
        every_token_ok: bool | None = None
        token_req_rate: float | None = None
        token_event_summary: dict[str, Any] = {
            "weighting": "event_weighted",
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "token_meet_rate": None,
            "excluded_or_unobservable_requests": offered,
        }
    elif offered == 0:
        token_status = "empty_population"
        every_token_ok = False
        token_req_rate = None
        token_event_summary = {
            "weighting": "event_weighted",
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "token_meet_rate": None,
            "excluded_or_unobservable_requests": 0,
        }
    elif (
        token_unavailable_requests > 0
        and token_evaluated_requests == 0
        and token_failed_or_invalid_requests == 0
    ):
        token_status = "unavailable"
        every_token_ok = None
        token_req_rate = None
        token_event_summary = {
            "weighting": "event_weighted",
            "evaluated_tokens": 0,
            "met_tokens": None,
            "missed_tokens": None,
            "token_meet_rate": None,
            "excluded_or_unobservable_requests": offered,
        }
    elif token_unavailable_requests > 0:
        token_status = (
            "partial_unavailable" if token_evaluated_requests > 0 else "unavailable"
        )
        every_token_ok = (
            False
            if (token_violating_requests > 0 or token_failed_or_invalid_requests > 0)
            else None
        )
        token_req_rate = None
        token_event_summary = {
            "weighting": "event_weighted",
            "evaluated_tokens": total_evaluated_tokens,
            "met_tokens": total_met_tokens if total_evaluated_tokens > 0 else None,
            "missed_tokens": (
                total_missed_tokens if total_evaluated_tokens > 0 else None
            ),
            "token_meet_rate": (
                total_met_tokens / total_evaluated_tokens
                if total_evaluated_tokens > 0
                else None
            ),
            "excluded_or_unobservable_requests": (
                token_unavailable_requests + token_failed_or_invalid_requests
            ),
        }
    else:
        every_token_ok = (
            token_compliant_requests == offered
            and token_failed_or_invalid_requests == 0
        )
        token_status = (
            "fail_closed_incomplete_population"
            if token_failed_or_invalid_requests > 0
            else ("passed" if every_token_ok else "failed")
        )
        token_req_rate = token_compliant_requests / offered
        token_event_summary = {
            "weighting": "event_weighted",
            "evaluated_tokens": total_evaluated_tokens,
            "met_tokens": total_met_tokens if total_evaluated_tokens > 0 else None,
            "missed_tokens": (
                total_missed_tokens if total_evaluated_tokens > 0 else None
            ),
            "token_meet_rate": (
                total_met_tokens / total_evaluated_tokens
                if total_evaluated_tokens > 0
                else None
            ),
            "excluded_or_unobservable_requests": token_failed_or_invalid_requests,
        }

    cumulative_token_deadline_compliance = {
        "status": token_status,
        "ttft_slo_seconds": ttft_slo,
        "tbt_slo_seconds": tbt_slo,
        "mapping_provenances": sorted(token_provenances),
        "denominator_requests": offered,
        "evaluated_requests": token_evaluated_requests,
        "compliant_requests": token_compliant_requests,
        "violating_requests": token_violating_requests,
        "unavailable_requests": token_unavailable_requests,
        "failed_or_invalid_requests": token_failed_or_invalid_requests,
        "request_weighted_compliance_rate": token_req_rate,
        "event_weighted_token_summary": token_event_summary,
        "every_token_meets_deadline": every_token_ok,
    }

    # Fail-closed run status.
    if offered == 0:
        run_status = "failed_empty_population"
    elif completed < offered:
        run_status = "failed_population"
    elif legacy_slo_met is False:
        run_status = "failed_slo"
    elif max_gap_slo_compliance["every_request_meets_max_gap_slo"] is False:
        run_status = "failed_slo"
    elif cumulative_token_deadline_compliance["every_token_meets_deadline"] is False:
        run_status = "failed_slo"
    elif (
        ttft_slo is not None
        and tbt_slo is not None
        and token_status in ("unavailable", "partial_unavailable")
    ):
        run_status = "incomplete_slo_unavailable"
    else:
        run_status = "passed"

    valid_duration = (
        duration_seconds
        if duration_seconds is not None
        and _is_finite_number(duration_seconds)
        and duration_seconds > 0
        else None
    )
    completed_rps = completed / valid_duration if valid_duration is not None else None
    max_gap_goodput = (
        max_gap_compliant / valid_duration
        if valid_duration is not None and max_gap_slo is not None
        else None
    )
    token_deadline_goodput = (
        token_compliant_requests / valid_duration
        if valid_duration is not None
        and ttft_slo is not None
        and tbt_slo is not None
        and token_unavailable_requests == 0
        else None
    )
    if valid_duration is None:
        goodput_rps = None
    elif ttft_slo is not None and tbt_slo is not None:
        if token_unavailable_requests > 0:
            goodput_rps = None
        else:
            strict_compliant = 0
            for record in records:
                if record.get("outcome") != "completed":
                    continue
                obs = record.get("stream_observation") or {}
                tdc_ok = (obs.get("token_deadline_compliance") or {}).get(
                    "every_token_meets_deadline"
                ) is True
                mg_ok = (
                    max_gap_slo is None
                    or (obs.get("max_gap_compliance") or {}).get("max_gap_within_budget")
                    is True
                )
                if tdc_ok and mg_ok:
                    strict_compliant += 1
            goodput_rps = strict_compliant / valid_duration
    elif max_gap_slo is not None:
        goodput_rps = max_gap_goodput
    else:
        goodput_rps = completed_rps if legacy_slo_met is not False else 0.0

    timestamp_domain = (
        TIMESTAMP_DOMAIN_NON_STREAM
        if any_non_stream and not any_sse
        else TIMESTAMP_DOMAIN_SSE
    )

    stream_deadlines = {
        "schema": STREAM_DEADLINE_SCHEMA,
        "timestamp_domain": timestamp_domain,
        "generation_time_attribution": GENERATION_TIME_ATTRIBUTION,
        "event_weighted": {
            "weighting": "event_weighted",
            "sample_count": len(all_event_gaps),
            "request_count": gap_request_count,
            "single_event_request_count": single_event_request_count,
            "excluded_or_unobservable_requests": excluded_gap_requests,
            "event_gap_seconds_p50": percentile(all_event_gaps, 0.50),
            "event_gap_seconds_p95": percentile(all_event_gaps, 0.95),
            "event_gap_seconds_p99": percentile(all_event_gaps, 0.99),
            "event_gap_seconds_max": max(all_event_gaps) if all_event_gaps else None,
        },
        "request_weighted": {
            "weighting": "request_weighted",
            "sample_count": len(per_request_max_gaps),
            "total_offered_requests": offered,
            "single_event_request_count": single_event_request_count,
            "excluded_or_unobservable_requests": excluded_gap_requests,
            "per_request_max_gap_seconds_p50": percentile(per_request_max_gaps, 0.50),
            "per_request_max_gap_seconds_p95": percentile(per_request_max_gaps, 0.95),
            "per_request_max_gap_seconds_p99": percentile(per_request_max_gaps, 0.99),
            "per_request_max_gap_seconds_max": (
                max(per_request_max_gaps) if per_request_max_gaps else None
            ),
            "per_request_p95_gap_seconds_p50": percentile(per_request_p95_gaps, 0.50),
            "per_request_p95_gap_seconds_p95": percentile(per_request_p95_gaps, 0.95),
            "per_request_p95_gap_seconds_max": (
                max(per_request_p95_gaps) if per_request_p95_gaps else None
            ),
        },
        "max_gap_slo_compliance": max_gap_slo_compliance,
        "cumulative_token_deadline_compliance": cumulative_token_deadline_compliance,
    }

    population_outcomes = {
        "offered": offered,
        "admitted": "unknown",
        "completed": completed,
        "rejected": rejected,
        "timed_out": timed_out,
        "incomplete": incomplete,
        "unobservable": unobservable,
        "cancelled": cancelled_count,
        "outcome_reasons": dict(sorted(reason_counts.items())),
        "completed_rate": completed / offered if offered > 0 else None,
        "failure_rate": (
            (rejected + timed_out + incomplete) / offered if offered > 0 else None
        ),
        "unknown_or_unobservable_rate": (
            unobservable / offered if offered > 0 else None
        ),
        "completed_requests_per_second": completed_rps,
        "goodput_requests_per_second": goodput_rps,
        "max_gap_goodput_requests_per_second": max_gap_goodput,
        "token_deadline_goodput_requests_per_second": token_deadline_goodput,
        "arrival_mode": arrival_mode,
        "duration_seconds": duration_seconds,
        "offered_load": offered_load,
        "warmup_runs": warmup_runs,
        "slo_denominators": {
            "offered_requests": offered,
            "max_gap_slo_denominator_requests": (
                offered if max_gap_slo is not None else None
            ),
            "token_deadline_denominator_requests": (
                offered if ttft_slo is not None and tbt_slo is not None else None
            ),
            "token_deadline_evaluated_tokens": (
                total_evaluated_tokens
                if ttft_slo is not None
                and tbt_slo is not None
                and token_unavailable_requests == 0
                else None
            ),
        },
        "run_status": run_status,
    }

    return {
        "population_outcomes": population_outcomes,
        "stream_deadlines": stream_deadlines,
        "run_status": run_status,
    }


def run_single(
    endpoint: Endpoint,
    model: str,
    prompts: list[str],
    max_tokens: int,
    stream: bool,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, prompt in enumerate(prompts):
        connection = open_connection(endpoint)
        try:
            record = perform_request(
                connection,
                endpoint,
                model,
                prompt,
                max_tokens,
                stream,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=deadline_slo,
            )
        finally:
            connection.close()
        record["prompt_index"] = index
        record["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        record["start_offset_seconds"] = 0.0
        records.append(record)
    return records


def run_batch_with_offsets(
    endpoint: Endpoint,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    stream: bool,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    """Batch with a shared barrier and per-record offsets from the group start."""
    connections: list[http.client.HTTPConnection] = []
    try:
        for _ in range(concurrency):
            connections.append(open_connection(endpoint))
    except BaseException:
        for connection in connections:
            connection.close()
        raise
    barrier = threading.Barrier(concurrency)
    records: list[dict[str, Any] | None] = [None] * concurrency
    starts: list[float | None] = [None] * concurrency
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=endpoint.timeout)
        except BaseException as error:  # noqa: BLE001 - fail the whole group.
            errors.append(error)
            barrier.abort()
            return
        prompt_index = index % len(prompts)
        try:
            start = clock()
            starts[index] = start
            record = perform_request(
                connections[index],
                endpoint,
                model,
                prompts[prompt_index],
                max_tokens,
                stream,
                start=start,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=deadline_slo,
            )
            record["prompt_index"] = prompt_index
            record["prompt_sha256"] = hashlib.sha256(
                prompts[prompt_index].encode()
            ).hexdigest()
            record["_start"] = start
            records[index] = record
        except BaseException as error:  # noqa: BLE001 - report every group member.
            errors.append(error)
        finally:
            connections[index].close()

    threads = [
        threading.Thread(target=worker, args=(index,), name=f"bench-{index}")
        for index in range(concurrency)
    ]
    started: list[threading.Thread] = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException:
        barrier.abort()
        for thread in started:
            thread.join()
        for connection in connections:
            connection.close()
        raise
    for thread in threads:
        thread.join()
    for connection in connections:
        connection.close()

    if errors:
        raise BenchmarkError(f"{endpoint.label}: batch failed: {errors[0]}")
    complete = [record for record in records if record is not None]
    if len(complete) != concurrency:
        raise BenchmarkError(
            f"{endpoint.label}: batch returned {len(complete)}/{concurrency} records"
        )
    start_values = [start for start in starts if start is not None]
    if len(start_values) != concurrency:
        raise BenchmarkError(f"{endpoint.label}: batch start timestamps are incomplete")
    group_start = min(start_values)
    for record in complete:
        record["start_offset_seconds"] = record.pop("_start") - group_start
    return complete


def summarize_records(
    mode: str,
    records: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    *,
    warmup_runs: int = 0,
    deadline_slo: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    ttfts = [
        record["ttft_seconds"]
        for record in records
        if record["ttft_seconds"] is not None
    ]
    walls = [
        record["wall_seconds"]
        for record in records
        if record["wall_seconds"] is not None
    ]
    completion_tokens = sum(
        record["completion_tokens"]
        for record in records
        if record["completion_tokens"] is not None
    )
    summary: dict[str, Any] = {
        "mode": mode,
        "requests": len(records),
        "completion_tokens": completion_tokens,
        "prompt_tokens": sum(record["prompt_tokens"] or 0 for record in records),
        "stream_chunks": sum(record["stream_chunks"] for record in records),
        "token_count_sources": sorted({
            record["token_count_source"] for record in records
        }),
        "ttft_seconds_median": median(ttfts),
        "ttft_seconds_p95": percentile(ttfts, 0.95),
        "wall_seconds_median": median(walls),
        "wall_seconds_p95": percentile(walls, 0.95),
        "wall_seconds_max": max(walls) if walls else None,
    }
    if mode.startswith("batch-"):
        summary["delivered_tokens_per_second"] = median([
            run["delivered_tokens_per_second"]
            for run in run_summaries
            if run["delivered_tokens_per_second"] is not None
        ])
        summary["aggregate_decode_tokens_per_second"] = median([
            run["aggregate_decode_tokens_per_second"]
            for run in run_summaries
            if run["aggregate_decode_tokens_per_second"] is not None
        ])
        summary["per_user_decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
    else:
        summary["delivered_tokens_per_second"] = (
            completion_tokens / sum(walls) if sum(walls) > 0 else None
        )
        summary["decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
        summary["e2e_tokens_per_second"] = median([
            record["e2e_tokens_per_second"]
            for record in records
            if record["e2e_tokens_per_second"] is not None
        ])

    if any("stream_observation" in record for record in records):
        if mode.startswith("batch-"):
            run_spans = [
                run["span_seconds"]
                for run in run_summaries
                if run.get("span_seconds") is not None
            ]
            mode_duration = sum(run_spans) if run_spans else None
            offered_load: dict[str, Any] = {
                "mode": mode,
                "concurrency": int(mode.removeprefix("batch-")),
                "measured_runs": len(run_summaries),
                "offered_requests": len(records),
            }
        else:
            mode_duration = sum(walls) if walls else None
            offered_load = {
                "mode": mode,
                "concurrency": 1,
                "measured_runs": len(run_summaries),
                "offered_requests": len(records),
            }
        ext = summarize_population_and_deadlines(
            records,
            duration_seconds=mode_duration,
            arrival_mode="closed_loop",
            offered_load=offered_load,
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        )
        summary["population_outcomes"] = ext["population_outcomes"]
        summary["stream_deadlines"] = ext["stream_deadlines"]
        summary["run_status"] = ext["run_status"]
    return summary


def summarize_run(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [
        record["wall_seconds"]
        for record in records
        if record["wall_seconds"] is not None
    ]
    completion_tokens = sum(
        record["completion_tokens"]
        for record in records
        if record["completion_tokens"] is not None
    )
    summary: dict[str, Any] = {
        "completion_tokens": completion_tokens,
        "wall_seconds_max": max(walls) if walls else None,
    }
    if mode.startswith("batch-"):
        starts = [record["start_offset_seconds"] for record in records]
        ends = [
            record["start_offset_seconds"] + record["wall_seconds"]
            for record in records
            if record["wall_seconds"] is not None
        ]
        span = max(ends) - min(starts) if starts and ends else 0.0
        summary["span_seconds"] = span
        summary["delivered_tokens_per_second"] = (
            completion_tokens / span if span > 0 else None
        )
        last_tokens = [
            record["start_offset_seconds"] + record["last_token_seconds"]
            for record in records
            if record["last_token_seconds"] is not None
        ]
        first_ttfts = [
            record["start_offset_seconds"] + record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ]
        summary["aggregate_decode_tokens_per_second"] = None
        if last_tokens and first_ttfts:
            last_token = max(last_tokens)
            first_ttft = min(first_ttfts)
            if last_token > first_ttft:
                summary["aggregate_decode_tokens_per_second"] = sum(
                    max((record["completion_tokens"] or 0) - 1, 0)
                    for record in records
                ) / (last_token - first_ttft)
        summary["per_user_decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
        summary["ttft_seconds_median"] = median([
            record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ])
    else:
        summary["delivered_tokens_per_second"] = (
            completion_tokens / sum(walls) if sum(walls) > 0 else None
        )
        summary["ttft_seconds_median"] = median([
            record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ])
    return summary


def agentic_messages(trace: dict[str, Any], index: int) -> list[dict[str, str]]:
    """Build the chat messages for one trace call.

    Each call's message list extends the previous call's list by the frozen
    assistant turn, the frozen tool result, and the next user turn, so the
    previous request text is a strict prefix of the current request text.
    """
    calls = trace["calls"]
    messages = [{"role": "system", "content": trace["system_prompt"]}]
    for call in calls[:index]:
        messages.append({"role": "user", "content": call["user_message"]})
        messages.append({"role": "assistant", "content": call["assistant_history"]})
        messages.append({
            "role": "user",
            "content": "Tool result:\n" + call["tool_result"],
        })
    messages.append({"role": "user", "content": calls[index]["user_message"]})
    return messages


def serialize_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}:{message['content']}" for message in messages)


def common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def validate_agentic_trace(trace: Any, source: str) -> dict[str, Any]:
    _require(isinstance(trace, dict), f"{source}: trace is not an object")
    _require(
        trace.get("schema_version") == 1,
        f"{source}: trace schema_version is not 1",
    )
    trace_id = trace.get("id")
    _require(
        isinstance(trace_id, str) and bool(trace_id.strip()),
        f"{source}: trace id is invalid",
    )
    system_prompt = trace.get("system_prompt")
    _require(
        isinstance(system_prompt, str) and bool(system_prompt.strip()),
        f"{source}: trace system prompt is invalid",
    )
    calls = trace.get("calls")
    _require(
        isinstance(calls, list) and len(calls) >= 2,
        f"{source}: trace must contain at least two calls",
    )
    previous = ""
    for index, call in enumerate(calls):
        _require(
            isinstance(call, dict), f"{source}: trace call {index} is not an object"
        )
        for name in ("user_message", "assistant_history", "tool_result"):
            value = call.get(name)
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{source}: trace call {index} {name} is invalid",
            )
        _require(
            _is_int(call.get("max_tokens")) and call["max_tokens"] > 0,
            f"{source}: trace call {index} max_tokens is invalid",
        )
        wait = call.get("tool_wait_seconds")
        _require(
            _is_finite_number(wait) and wait >= 0,
            f"{source}: trace call {index} tool_wait_seconds is invalid",
        )
        serialized = serialize_messages(agentic_messages(trace, index))
        _require(
            serialized.startswith(previous),
            f"{source}: trace call {index} does not extend the previous history",
        )
        previous = serialized
    return trace


def load_agentic_trace(path: Path) -> dict[str, Any]:
    try:
        trace = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read trace {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"trace {path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"trace {path} is not JSON: {error}") from error
    return validate_agentic_trace(trace, str(path))


def agentic_trace_sha256(trace: dict[str, Any]) -> str:
    return canonical_sha256(trace)


def agentic_trace_workload(
    trace: dict[str, Any], tool_wait_scale: float
) -> dict[str, Any]:
    calls = []
    total_wait = 0.0
    for index, call in enumerate(trace["calls"]):
        scaled_wait = call["tool_wait_seconds"] * tool_wait_scale
        total_wait += scaled_wait
        calls.append({
            "index": index,
            "messages_sha256": hashlib.sha256(
                serialize_messages(agentic_messages(trace, index)).encode()
            ).hexdigest(),
            "max_tokens": call["max_tokens"],
            "tool_wait_seconds": call["tool_wait_seconds"],
            "tool_wait_seconds_scaled": scaled_wait,
        })
    return {
        "trace_id": trace["id"],
        "trace_sha256": agentic_trace_sha256(trace),
        "system_prompt_sha256": hashlib.sha256(
            trace["system_prompt"].encode()
        ).hexdigest(),
        "calls": calls,
        "tool_wait_scale": tool_wait_scale,
        "tool_wait_seconds_total": total_wait,
    }


def _total_or_none(records: list[dict[str, Any]], key: str) -> int | float | None:
    values = [record[key] for record in records]
    if any(value is None for value in values):
        return None
    return sum(values)


def agentic_call_record(
    record: dict[str, Any],
    usage: dict[str, Any] | None,
    agent_index: int,
    call_index: int,
    start_offset: float,
    tool_wait_seconds: float,
    max_tokens: int,
    reused_characters: int,
    prompt_characters: int,
    messages_sha256: str,
) -> dict[str, Any]:
    """Add agentic accounting to a measured chat call.

    Reused-prefix tokens are a client-side apportionment of the endpoint's
    reported prompt_tokens by the exact client-known prefix character fraction.
    They never claim that a server kept those tokens in a cache.
    """
    prompt_tokens = record.get("prompt_tokens")
    fraction = reused_characters / prompt_characters if prompt_characters > 0 else 0.0
    reused_estimate = None
    new_estimate = None
    if prompt_tokens is not None:
        reused_estimate = round(prompt_tokens * fraction)
        new_estimate = prompt_tokens - reused_estimate
    cached_tokens = None
    cached_source = "unreported"
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and _is_int(details.get("cached_tokens")):
            cached_tokens = details["cached_tokens"]
            cached_source = "usage.prompt_tokens_details.cached_tokens"
    ttft = record.get("ttft_seconds")
    last = record.get("last_token_seconds")
    completion_tokens = record.get("completion_tokens")
    itl = None
    if (
        record.get("outcome", "completed") == "completed"
        and ttft is not None
        and last is not None
        and last > ttft
        and completion_tokens is not None
        and completion_tokens > 1
    ):
        itl = (last - ttft) / (completion_tokens - 1)
    record.update({
        "agent_index": agent_index,
        "call_index": call_index,
        "messages_sha256": messages_sha256,
        "max_tokens": max_tokens,
        "start_offset_seconds": start_offset,
        "tool_wait_after_seconds": tool_wait_seconds,
        "prompt_characters": prompt_characters,
        "reused_prefix_characters": reused_characters,
        "reused_prefix_fraction": fraction,
        "reused_prefix_tokens_estimate": reused_estimate,
        "new_input_tokens_estimate": new_estimate,
        "server_reported_cached_tokens": cached_tokens,
        "server_reported_cached_tokens_source": cached_source,
        "cache_confirmation": "client-prefix-overlap-only",
        "itl_seconds_mean": itl,
    })
    return record


def _agentic_worker(
    endpoint: Endpoint,
    model: str,
    trace: dict[str, Any],
    agent_index: int,
    tool_wait_scale: float,
    barrier: threading.Barrier,
    results: list[dict[str, Any] | None],
    errors: list[BaseException],
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
) -> None:
    try:
        barrier.wait(timeout=endpoint.timeout)
    except BaseException as error:  # noqa: BLE001 - fail the whole group.
        errors.append(error)
        barrier.abort()
        return
    calls = trace["calls"]
    task_start = clock()
    previous_serialized = ""
    records: list[dict[str, Any]] = []
    total_wait = 0.0
    try:
        for index, call in enumerate(calls):
            messages = agentic_messages(trace, index)
            serialized = serialize_messages(messages)
            reused = common_prefix_length(serialized, previous_serialized)
            previous_serialized = serialized
            wait = call["tool_wait_seconds"] * tool_wait_scale
            call_start = clock()
            connection: http.client.HTTPConnection | None = None
            try:
                connection = open_connection(endpoint)
                call_start = clock()
                record, usage = observe_stream_completion(
                    connection,
                    endpoint,
                    endpoint.api_path("/chat/completions"),
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": call["max_tokens"],
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    f"agent {agent_index} call {index}",
                    call_start,
                    clock=clock,
                    max_trace_events=max_trace_events,
                    deadline_slo=deadline_slo,
                )
            except (TimeoutError, socket.timeout) as error:
                record = _failure_stream_record(
                    outcome="timed_out",
                    outcome_reason="connect_timeout",
                    error_message=f"{endpoint.label}: connect timeout: {error}",
                    start=call_start,
                    end=clock(),
                    max_trace_events=max_trace_events,
                    deadline_slo=deadline_slo,
                )
                usage = None
            except ConnectionRefusedError as error:
                record = _failure_stream_record(
                    outcome="rejected",
                    outcome_reason="connection_refused",
                    error_message=f"{endpoint.label}: connection refused: {error}",
                    start=call_start,
                    end=clock(),
                    max_trace_events=max_trace_events,
                    deadline_slo=deadline_slo,
                )
                usage = None
            except (OSError, http.client.HTTPException) as error:
                record = _failure_stream_record(
                    outcome="incomplete",
                    outcome_reason=f"connect_{type(error).__name__}",
                    error_message=(
                        f"{endpoint.label}: connect error: {type(error).__name__}: {error}"
                    ),
                    start=call_start,
                    end=clock(),
                    max_trace_events=max_trace_events,
                    deadline_slo=deadline_slo,
                )
                usage = None
            finally:
                if connection is not None:
                    connection.close()
            records.append(
                agentic_call_record(
                    record,
                    usage,
                    agent_index,
                    index,
                    max(0.0, call_start - task_start),
                    wait,
                    call["max_tokens"],
                    reused,
                    len(serialized),
                    hashlib.sha256(serialized.encode()).hexdigest(),
                )
            )
            if wait > 0:
                time.sleep(wait)
            total_wait += wait
        task_end = clock()
        task_outcome = (
            "completed"
            if all(item.get("outcome", "completed") == "completed" for item in records)
            else "failed"
        )
        results[agent_index] = {
            "agent_index": agent_index,
            "call_count": len(calls),
            "task_outcome": task_outcome,
            "task_seconds": max(task_end - task_start, 0.0),
            "start_offset_seconds": 0.0,
            "tool_wait_seconds": total_wait,
            "prompt_tokens": _total_or_none(records, "prompt_tokens"),
            "completion_tokens": _total_or_none(records, "completion_tokens"),
            "reused_prefix_characters": _total_or_none(
                records, "reused_prefix_characters"
            ),
            "reused_prefix_tokens_estimate": _total_or_none(
                records, "reused_prefix_tokens_estimate"
            ),
            "new_input_tokens_estimate": _total_or_none(
                records, "new_input_tokens_estimate"
            ),
            "server_reported_cached_tokens": _total_or_none(
                records, "server_reported_cached_tokens"
            ),
            "calls": records,
            "_task_start": task_start,
        }
    except BaseException as error:  # noqa: BLE001 - report every agent failure.
        errors.append(error)


def run_agentic_group(
    endpoint: Endpoint,
    model: str,
    trace: dict[str, Any],
    agents: int,
    tool_wait_scale: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    """Run ``agents`` closed-loop traces released from one barrier."""
    barrier = threading.Barrier(agents)
    results: list[dict[str, Any] | None] = [None] * agents
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_agentic_worker,
            args=(
                endpoint,
                model,
                trace,
                agent_index,
                tool_wait_scale,
                barrier,
                results,
                errors,
            ),
            kwargs={
                "clock": clock,
                "max_trace_events": max_trace_events,
                "deadline_slo": deadline_slo,
            },
            name=f"agent-{agent_index}",
        )
        for agent_index in range(agents)
    ]
    started: list[threading.Thread] = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException:
        barrier.abort()
        for thread in started:
            thread.join()
        raise
    for thread in threads:
        thread.join()
    if errors:
        raise BenchmarkError(f"{endpoint.label}: agentic group failed: {errors[0]}")
    tasks = [task for task in results if task is not None]
    if len(tasks) != agents:
        raise BenchmarkError(
            f"{endpoint.label}: agentic group returned {len(tasks)}/{agents} tasks"
        )
    group_start = min(task["_task_start"] for task in tasks)
    for task in tasks:
        start = task.pop("_task_start")
        task["start_offset_seconds"] = start - group_start
    return tasks


def agentic_slo_flags(
    summary: dict[str, Any],
    slo: dict[str, float],
    *,
    all_completed: bool = True,
    max_gap_met: bool | None = None,
    token_deadline_met: bool | None = None,
    has_token_deadline_slo: bool = False,
) -> dict[str, Any]:
    ttft = summary["ttft_seconds_p95"]
    itl = summary["itl_seconds_p95"]
    task = summary["task_seconds_p95"]
    meets_ttft = all_completed and ttft is not None and ttft <= slo["ttft_p95_seconds"]
    meets_itl = all_completed and itl is not None and itl <= slo["itl_p95_seconds"]
    meets_task = all_completed and task is not None and task <= slo["task_p95_seconds"]
    meets_all = meets_ttft and meets_itl and meets_task
    if max_gap_met is not None:
        meets_all = meets_all and bool(max_gap_met)
    if has_token_deadline_slo:
        meets_all = meets_all and (token_deadline_met is True)
    flags: dict[str, Any] = {
        "meets_ttft_slo": meets_ttft,
        "meets_itl_slo": meets_itl,
        "meets_task_slo": meets_task,
        "meets_all_slo": meets_all,
    }
    if "stream_deadlines" in summary:
        flags["meets_max_gap_slo"] = max_gap_met
        flags["meets_token_deadline_slo"] = token_deadline_met
    return flags


def summarize_agentic_run(
    agents: int,
    tasks: list[dict[str, Any]],
    slo: dict[str, float],
    *,
    warmup_runs: int = 0,
    deadline_slo: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    calls = [call for task in tasks for call in task["calls"]]
    has_stream_ext = any("stream_observation" in call for call in calls)
    completed_calls = [
        call for call in calls if call.get("outcome", "completed") == "completed"
    ]
    completed_tasks = [
        task
        for task in tasks
        if all(call.get("outcome", "completed") == "completed" for call in task["calls"])
    ]
    all_completed = len(completed_calls) == len(calls) and len(calls) > 0
    starts = [task["start_offset_seconds"] for task in tasks]
    ends = [task["start_offset_seconds"] + task["task_seconds"] for task in tasks]
    span = max(ends) - min(starts) if starts else 0.0
    output_tokens = _total_or_none(completed_calls, "completion_tokens") if all_completed else None
    task_seconds = [task["task_seconds"] for task in completed_tasks]
    summary: dict[str, Any] = {
        "agents": agents,
        "tasks": len(tasks),
        "calls": len(calls),
        "task_seconds_median": median(task_seconds),
        "task_seconds_p95": percentile(task_seconds, 0.95),
        "task_seconds_max": max(task_seconds) if task_seconds else None,
        "ttft_seconds_median": median([
            call["ttft_seconds"]
            for call in completed_calls
            if call["ttft_seconds"] is not None
        ]),
        "ttft_seconds_p95": percentile(
            [
                call["ttft_seconds"]
                for call in completed_calls
                if call["ttft_seconds"] is not None
            ],
            0.95,
        ),
        "itl_seconds_median": median([
            call["itl_seconds_mean"]
            for call in completed_calls
            if call["itl_seconds_mean"] is not None
        ]),
        "itl_seconds_p95": percentile(
            [
                call["itl_seconds_mean"]
                for call in completed_calls
                if call["itl_seconds_mean"] is not None
            ],
            0.95,
        ),
        "tasks_per_second": len(completed_tasks) / span if span > 0 else None,
        "output_tokens_per_second": (
            output_tokens / span if span > 0 and output_tokens is not None else None
        ),
    }
    if has_stream_ext:
        summary["span_seconds"] = span
        legacy_ok = (
            all_completed
            and summary["ttft_seconds_p95"] is not None
            and summary["ttft_seconds_p95"] <= slo["ttft_p95_seconds"]
            and summary["itl_seconds_p95"] is not None
            and summary["itl_seconds_p95"] <= slo["itl_p95_seconds"]
            and summary["task_seconds_p95"] is not None
            and summary["task_seconds_p95"] <= slo["task_p95_seconds"]
        )
        ext = summarize_population_and_deadlines(
            calls,
            duration_seconds=span,
            arrival_mode="closed_loop",
            offered_load={
                "agents": agents,
                "tasks": len(tasks),
                "calls_per_task": len(tasks[0]["calls"]) if tasks else 0,
                "offered_requests": len(calls),
            },
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
            legacy_slo_met=legacy_ok,
        )
        summary["population_outcomes"] = ext["population_outcomes"]
        summary["stream_deadlines"] = ext["stream_deadlines"]
        summary["run_status"] = ext["run_status"]
        norm_slo = _validate_deadline_slo(deadline_slo)
        max_gap_met = ext["stream_deadlines"]["max_gap_slo_compliance"][
            "every_request_meets_max_gap_slo"
        ]
        token_met = ext["stream_deadlines"]["cumulative_token_deadline_compliance"][
            "every_token_meets_deadline"
        ]
        summary.update(
            agentic_slo_flags(
                summary,
                slo,
                all_completed=all_completed,
                max_gap_met=max_gap_met,
                token_deadline_met=token_met,
                has_token_deadline_slo=(
                    norm_slo["ttft_slo_seconds"] is not None
                    and norm_slo["tbt_slo_seconds"] is not None
                ),
            )
        )
    else:
        summary.update(agentic_slo_flags(summary, slo, all_completed=all_completed))
    return summary


def summarize_agentic_width(
    agents: int,
    tasks: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    slo: dict[str, float],
    *,
    warmup_runs: int = 0,
    deadline_slo: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    calls = [call for task in tasks for call in task["calls"]]
    has_stream_ext = any("stream_observation" in call for call in calls)
    completed_calls = [
        call for call in calls if call.get("outcome", "completed") == "completed"
    ]
    completed_tasks = [
        task
        for task in tasks
        if all(call.get("outcome", "completed") == "completed" for call in task["calls"])
    ]
    all_completed = len(completed_calls) == len(calls) and len(calls) > 0
    task_seconds = [task["task_seconds"] for task in completed_tasks]
    ttfts = [
        call["ttft_seconds"]
        for call in completed_calls
        if call["ttft_seconds"] is not None
    ]
    itls = [
        call["itl_seconds_mean"]
        for call in completed_calls
        if call["itl_seconds_mean"] is not None
    ]
    summary: dict[str, Any] = {
        "agents": agents,
        "tasks": len(tasks),
        "calls": len(calls),
        "task_seconds_median": median(task_seconds),
        "task_seconds_p95": percentile(task_seconds, 0.95),
        "task_seconds_max": max(task_seconds) if task_seconds else None,
        "ttft_seconds_median": median(ttfts),
        "ttft_seconds_p95": percentile(ttfts, 0.95),
        "itl_seconds_median": median(itls),
        "itl_seconds_p95": percentile(itls, 0.95),
        "output_tokens": _total_or_none(calls, "completion_tokens"),
        "prompt_tokens": _total_or_none(calls, "prompt_tokens"),
        "reused_prefix_characters": sum(
            call["reused_prefix_characters"] for call in calls
        ),
        "reused_prefix_tokens_estimate": _total_or_none(
            calls, "reused_prefix_tokens_estimate"
        ),
        "new_input_tokens_estimate": _total_or_none(calls, "new_input_tokens_estimate"),
        "server_reported_cached_tokens": _total_or_none(
            calls, "server_reported_cached_tokens"
        ),
        "tasks_per_second": median([
            run["tasks_per_second"]
            for run in run_summaries
            if run["tasks_per_second"] is not None
        ]),
        "output_tokens_per_second": median([
            run["output_tokens_per_second"]
            for run in run_summaries
            if run["output_tokens_per_second"] is not None
        ]),
        "agents_at_slo_per_watt": None,
    }
    if has_stream_ext:
        spans = [
            run["span_seconds"]
            for run in run_summaries
            if run.get("span_seconds") is not None
        ]
        width_duration = sum(spans) if spans else None
        legacy_ok = (
            all_completed
            and summary["ttft_seconds_p95"] is not None
            and summary["ttft_seconds_p95"] <= slo["ttft_p95_seconds"]
            and summary["itl_seconds_p95"] is not None
            and summary["itl_seconds_p95"] <= slo["itl_p95_seconds"]
            and summary["task_seconds_p95"] is not None
            and summary["task_seconds_p95"] <= slo["task_p95_seconds"]
        )
        ext = summarize_population_and_deadlines(
            calls,
            duration_seconds=width_duration,
            arrival_mode="closed_loop",
            offered_load={
                "agents": agents,
                "measured_runs": len(run_summaries),
                "tasks": len(tasks),
                "offered_requests": len(calls),
            },
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
            legacy_slo_met=legacy_ok,
        )
        summary["population_outcomes"] = ext["population_outcomes"]
        summary["stream_deadlines"] = ext["stream_deadlines"]
        summary["run_status"] = ext["run_status"]
        norm_slo = _validate_deadline_slo(deadline_slo)
        max_gap_met = ext["stream_deadlines"]["max_gap_slo_compliance"][
            "every_request_meets_max_gap_slo"
        ]
        token_met = ext["stream_deadlines"]["cumulative_token_deadline_compliance"][
            "every_token_meets_deadline"
        ]
        summary.update(
            agentic_slo_flags(
                summary,
                slo,
                all_completed=all_completed,
                max_gap_met=max_gap_met,
                token_deadline_met=token_met,
                has_token_deadline_slo=(
                    norm_slo["ttft_slo_seconds"] is not None
                    and norm_slo["tbt_slo_seconds"] is not None
                ),
            )
        )
    else:
        summary.update(agentic_slo_flags(summary, slo, all_completed=all_completed))
    return summary


def run_agentic_benchmark(
    endpoint: Endpoint,
    trace: dict[str, Any],
    agents: list[int],
    runs: int,
    warmup_runs: int,
    settle_seconds: float,
    tool_wait_scale: float,
    slo: dict[str, float],
    cache_mode: str,
    power: dict[str, Any],
    revision: str | None,
    note: str | None,
    *,
    deadline_slo: dict[str, float | None] | None = None,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    validate_agentic_trace(trace, "trace")
    if not _is_int(max_trace_events) or max_trace_events <= 0:
        raise BenchmarkError("max_trace_events must be a positive integer")
    norm_deadline_slo = _validate_deadline_slo(deadline_slo)
    if set(slo) != {"ttft_p95_seconds", "itl_p95_seconds", "task_p95_seconds"}:
        raise BenchmarkError(
            "SLO keys must be ttft_p95_seconds, itl_p95_seconds, task_p95_seconds"
        )
    if (
        not agents
        or any(not _is_int(width) or width < 1 for width in agents)
        or len(set(agents)) != len(agents)
    ):
        raise BenchmarkError("agent widths must be unique integers greater than zero")
    if not _is_int(runs) or runs <= 0 or not _is_int(warmup_runs) or warmup_runs < 0:
        raise BenchmarkError("runs must be positive; warmup_runs non-negative")
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise BenchmarkError("settle seconds must be finite and non-negative")
    if not math.isfinite(tool_wait_scale) or tool_wait_scale < 0:
        raise BenchmarkError("tool wait scale must be finite and non-negative")
    for name, value in slo.items():
        if not _is_finite_number(value) or value <= 0:
            raise BenchmarkError(f"SLO {name} must be finite and positive")
    if cache_mode not in CACHE_MODES:
        raise BenchmarkError(f"cache mode must be one of {', '.join(CACHE_MODES)}")
    boundary = power.get("boundary")
    watts = power.get("watts")
    method = power.get("method")
    if boundary not in POWER_BOUNDARIES:
        raise BenchmarkError(
            f"power boundary must be one of {', '.join(POWER_BOUNDARIES)}"
        )
    if watts is not None and (not _is_finite_number(watts) or watts <= 0):
        raise BenchmarkError("power watts must be finite and positive")
    if watts is not None and boundary == "unspecified":
        raise BenchmarkError("--watts requires a concrete --power-boundary")
    if boundary != "unspecified" and watts is None:
        raise BenchmarkError("--power-boundary requires --watts")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise BenchmarkError("power method must be a non-empty string")

    workload = agentic_trace_workload(trace, tool_wait_scale)
    workload.update({
        "agents": agents,
        "measured_runs": runs,
        "warmup_runs": warmup_runs,
        "settle_seconds": settle_seconds,
        "temperature": 0,
        "stream": True,
        "cache_mode": cache_mode,
    })
    workload_sha256 = canonical_sha256(workload)
    trace_sha256 = workload["trace_sha256"]
    model, probe = resolve_model(endpoint)

    started = utc_now()
    started_monotonic = clock()
    for _ in range(warmup_runs):
        for width in agents:
            run_agentic_group(
                endpoint,
                model,
                trace,
                width,
                tool_wait_scale,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=norm_deadline_slo,
            )
    if warmup_runs:
        time.sleep(settle_seconds)

    measured: list[dict[str, Any]] = []
    for run_index in range(runs):
        for width in agents:
            tasks = run_agentic_group(
                endpoint,
                model,
                trace,
                width,
                tool_wait_scale,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=norm_deadline_slo,
            )
            measured.append({
                "agents": width,
                "index": run_index,
                "summary": summarize_agentic_run(
                    width,
                    tasks,
                    slo,
                    warmup_runs=warmup_runs,
                    deadline_slo=norm_deadline_slo,
                ),
                "tasks": tasks,
            })
            time.sleep(settle_seconds)

    curve: list[dict[str, Any]] = []
    for width in agents:
        width_tasks = [
            task for run in measured if run["agents"] == width for task in run["tasks"]
        ]
        width_runs = [run["summary"] for run in measured if run["agents"] == width]
        point = summarize_agentic_width(
            width,
            width_tasks,
            width_runs,
            slo,
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        )
        if watts is not None and point["meets_all_slo"]:
            point["agents_at_slo_per_watt"] = width / watts
        curve.append(point)

    all_calls = [
        call for run in measured for task in run["tasks"] for call in task["calls"]
    ]
    total_span = sum(
        run["summary"].get("span_seconds", 0.0) for run in measured
    )
    overall_ext = summarize_population_and_deadlines(
        all_calls,
        duration_seconds=total_span if total_span > 0 else None,
        arrival_mode="closed_loop",
        offered_load={
            "agents": agents,
            "measured_runs": runs,
            "calls_per_task": len(trace["calls"]),
            "offered_requests": len(all_calls),
        },
        warmup_runs=warmup_runs,
        deadline_slo=norm_deadline_slo,
        legacy_slo_met=any(point["meets_all_slo"] for point in curve),
    )

    meeting = [point["agents"] for point in curve if point["meets_all_slo"]]
    max_agents = max(meeting) if meeting else None
    throughput_points = [
        point for point in curve if point["tasks_per_second"] is not None
    ]
    saturated = (
        max(throughput_points, key=lambda point: point["tasks_per_second"])["agents"]
        if throughput_points
        else None
    )
    slo_point = (
        next((point for point in curve if point["agents"] == max_agents), None)
        if max_agents is not None
        else None
    )
    power_record = {
        "boundary": boundary,
        "watts": watts,
        "method": method,
        "agents_at_slo_per_watt": (
            max_agents / watts if max_agents is not None and watts is not None else None
        ),
        "output_tokens_per_second_per_watt": (
            slo_point["output_tokens_per_second"] / watts
            if slo_point is not None
            and watts is not None
            and slo_point["output_tokens_per_second"] is not None
            else None
        ),
    }

    finished = utc_now()
    return {
        "schema": AGENTIC_SCHEMA,
        "schema_extensions": [STREAM_DEADLINE_SCHEMA],
        "generated_utc": finished,
        "started_utc": started,
        "duration_seconds": clock() - started_monotonic,
        "note": note,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "trace": trace,
        "trace_sha256": trace_sha256,
        "target": {
            "label": endpoint.label,
            "base_url": endpoint.base_url,
            "model": model,
            "probe": probe,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": source_revision(revision),
            "selected_environment": selected_environment(),
        },
        "conditions": {
            "slo": slo,
            "deadline_slo": norm_deadline_slo,
            "max_trace_events": max_trace_events,
            "arrival_mode": "closed_loop",
            "cache_mode": cache_mode,
            "power": power_record,
            "temperature": 0,
            "stream": True,
            "request_timeout_seconds": endpoint.timeout,
        },
        "summary": {
            "slo": slo,
            "max_agents_meeting_all_slo": max_agents,
            "saturated_agents": saturated,
            "curve": curve,
            "population_outcomes": overall_ext["population_outcomes"],
            "stream_deadlines": overall_ext["stream_deadlines"],
            "run_status": overall_ext["run_status"],
        },
        "runs": measured,
    }


def _validate_stream_observation(
    record: dict[str, Any],
    source: str,
    deadline_slo: dict[str, float | None] | None = None,
) -> None:
    obs = record.get("stream_observation")
    _require(isinstance(obs, dict), f"{source}: stream_observation is not an object")
    _require(
        obs.get("schema") == STREAM_DEADLINE_SCHEMA,
        f"{source}: stream_observation schema is invalid",
    )
    _require(
        obs.get("generation_time_attribution") == GENERATION_TIME_ATTRIBUTION,
        f"{source}: stream_observation generation_time_attribution is invalid",
    )
    _require(
        obs.get("overflow_policy") == OVERFLOW_POLICY,
        f"{source}: stream_observation overflow_policy is invalid",
    )
    domain = obs.get("timestamp_domain")
    _require(
        domain in (TIMESTAMP_DOMAIN_SSE, TIMESTAMP_DOMAIN_NON_STREAM),
        f"{source}: stream_observation timestamp_domain is invalid",
    )
    max_events = obs.get("max_trace_events")
    recorded_events = obs.get("recorded_trace_events")
    dropped_events = obs.get("dropped_trace_events")
    truncated = obs.get("trace_truncated")
    overflow = obs.get("overflow")
    _require(
        _is_int(max_events) and max_events >= 0,
        f"{source}: stream_observation max_trace_events is invalid",
    )
    _require(
        _is_int(recorded_events) and recorded_events >= 0,
        f"{source}: stream_observation recorded_trace_events is invalid",
    )
    _require(
        _is_int(dropped_events) and dropped_events >= 0,
        f"{source}: stream_observation dropped_trace_events is invalid",
    )
    _require(
        isinstance(truncated, bool) and isinstance(overflow, bool) and truncated == overflow,
        f"{source}: stream_observation trace_truncated/overflow is invalid",
    )
    offsets = obs.get("content_event_offsets_seconds")
    _require(
        isinstance(offsets, list) and len(offsets) == recorded_events,
        f"{source}: stream_observation content_event_offsets_seconds length mismatch",
    )
    if domain == TIMESTAMP_DOMAIN_SSE:
        _require(
            max_events > 0 and recorded_events <= max_events,
            f"{source}: stream_observation exceeded max_trace_events bound",
        )
        if truncated:
            _require(
                dropped_events > 0,
                f"{source}: truncated stream must record dropped_trace_events > 0",
            )
            _require(
                (obs.get("content_event_gaps") or {}).get("status")
                == "invalid_truncated",
                f"{source}: truncated stream must invalidate content_event_gaps",
            )
            _require(
                (obs.get("max_gap_compliance") or {}).get("max_gap_within_budget")
                is not True,
                f"{source}: truncated stream must never pass max_gap_compliance",
            )
            _require(
                (obs.get("token_deadline_compliance") or {}).get(
                    "every_token_meets_deadline"
                )
                is not True,
                f"{source}: truncated stream must never pass token_deadline_compliance",
            )
    event_counts = obs.get("event_counts")
    _require(
        isinstance(event_counts, dict)
        and set(event_counts)
        == {"content", "heartbeat", "role_only", "usage_only", "empty", "done"}
        and all(_is_int(v) and v >= 0 for v in event_counts.values()),
        f"{source}: stream_observation event_counts is invalid",
    )
    timing_valid = obs.get("timing_valid")
    _require(
        isinstance(timing_valid, bool),
        f"{source}: stream_observation timing_valid is invalid",
    )
    if domain == TIMESTAMP_DOMAIN_SSE and not truncated and timing_valid:
        slo = _validate_deadline_slo(deadline_slo)
        recomputed = evaluate_stream_timestamps(
            offsets,
            ttft_slo_seconds=slo["ttft_slo_seconds"],
            tbt_slo_seconds=slo["tbt_slo_seconds"],
            max_gap_slo_seconds=slo["max_gap_slo_seconds"],
            trace_truncated=False,
            timing_valid=True,
        )
        _require(
            _summary_value_matches(
                recomputed["content_event_gaps"], obs.get("content_event_gaps")
            ),
            f"{source}: stream_observation content_event_gaps does not match offsets",
        )
        _require(
            _summary_value_matches(
                recomputed["max_gap_compliance"], obs.get("max_gap_compliance")
            ),
            f"{source}: stream_observation max_gap_compliance does not match offsets",
        )


def validate_agentic_result(result: Any, source: str = "result") -> dict[str, Any]:
    _require(isinstance(result, dict), f"{source}: result is not an object")
    _require(
        result.get("schema") == AGENTIC_SCHEMA,
        f"{source}: schema is not {AGENTIC_SCHEMA}",
    )
    target = result.get("target")
    _require(
        isinstance(target, dict)
        and isinstance(target.get("label"), str)
        and bool(target["label"]),
        f"{source}: target label is missing",
    )
    _require(
        isinstance(target.get("base_url"), str) and bool(target["base_url"]),
        f"{source}: target base_url is missing",
    )
    _require(
        isinstance(target.get("model"), str) and bool(target["model"]),
        f"{source}: target model is missing",
    )
    _require(
        isinstance(result.get("environment"), dict),
        f"{source}: environment is not an object",
    )

    trace = validate_agentic_trace(result.get("trace"), source)
    trace_sha256 = result.get("trace_sha256")
    _require(
        isinstance(trace_sha256, str)
        and len(trace_sha256) == 64
        and canonical_sha256(trace) == trace_sha256,
        f"{source}: trace digest does not match the trace",
    )
    workload = result.get("workload")
    _require(isinstance(workload, dict), f"{source}: workload is not an object")
    _require(
        isinstance(result.get("workload_sha256"), str)
        and canonical_sha256(workload) == result["workload_sha256"],
        f"{source}: workload digest does not match its workload",
    )
    _require(
        workload.get("trace_sha256") == trace_sha256,
        f"{source}: workload trace digest does not match the trace",
    )
    _require(
        workload.get("trace_id") == trace["id"],
        f"{source}: workload trace id does not match the trace",
    )
    agents = workload.get("agents")
    _require(
        isinstance(agents, list)
        and bool(agents)
        and all(_is_int(width) and width > 0 for width in agents)
        and len(set(agents)) == len(agents),
        f"{source}: workload agents are invalid",
    )
    measured_runs = workload.get("measured_runs")
    _require(
        _is_int(measured_runs) and measured_runs > 0,
        f"{source}: workload measured_runs is invalid",
    )
    warmup_runs = workload.get("warmup_runs")
    _require(
        _is_int(warmup_runs) and warmup_runs >= 0,
        f"{source}: workload warmup_runs is invalid",
    )
    _require(
        _is_finite_number(workload.get("settle_seconds"))
        and workload["settle_seconds"] >= 0,
        f"{source}: workload settle_seconds is invalid",
    )
    tool_wait_scale = workload.get("tool_wait_scale")
    _require(
        _is_finite_number(tool_wait_scale) and tool_wait_scale >= 0,
        f"{source}: workload tool_wait_scale is invalid",
    )
    _require(
        workload.get("temperature") == 0,
        f"{source}: workload temperature is not greedy",
    )
    _require(
        workload.get("stream") is True,
        f"{source}: workload stream flag is not true",
    )
    cache_mode = workload.get("cache_mode")
    _require(
        cache_mode in CACHE_MODES,
        f"{source}: workload cache mode is invalid",
    )
    expected_calls = agentic_trace_workload(trace, tool_wait_scale)["calls"]
    _require(
        workload.get("calls") == expected_calls,
        f"{source}: workload calls do not match the trace",
    )
    _require(
        _summary_value_matches(
            sum(call["tool_wait_seconds_scaled"] for call in expected_calls),
            workload.get("tool_wait_seconds_total"),
        ),
        f"{source}: workload tool wait total is invalid",
    )
    _require(
        workload.get("system_prompt_sha256")
        == hashlib.sha256(trace["system_prompt"].encode()).hexdigest(),
        f"{source}: workload system prompt digest is invalid",
    )

    conditions = result.get("conditions")
    _require(isinstance(conditions, dict), f"{source}: conditions are invalid")
    slo = conditions.get("slo")
    _require(isinstance(slo, dict), f"{source}: SLO is invalid")
    _require(
        set(slo) == {"ttft_p95_seconds", "itl_p95_seconds", "task_p95_seconds"},
        f"{source}: SLO keys are invalid",
    )
    for name, value in slo.items():
        _require(
            _is_finite_number(value) and value > 0,
            f"{source}: SLO {name} is invalid",
        )
    deadline_slo = (
        _validate_deadline_slo(conditions.get("deadline_slo"))
        if "deadline_slo" in conditions
        else None
    )
    _require(
        conditions.get("cache_mode") == cache_mode,
        f"{source}: condition cache mode does not match the workload",
    )
    _require(
        conditions.get("temperature") == 0 and conditions.get("stream") is True,
        f"{source}: conditions are not greedy streaming",
    )
    _require(
        _is_finite_number(conditions.get("request_timeout_seconds"))
        and conditions["request_timeout_seconds"] > 0,
        f"{source}: request timeout condition is invalid",
    )
    power = conditions.get("power")
    _require(isinstance(power, dict), f"{source}: power conditions are invalid")
    boundary = power.get("boundary")
    watts = power.get("watts")
    _require(
        boundary in POWER_BOUNDARIES,
        f"{source}: power boundary is invalid",
    )
    _require(
        watts is None or (_is_finite_number(watts) and watts > 0),
        f"{source}: power watts are invalid",
    )
    _require(
        (watts is None) == (boundary == "unspecified"),
        f"{source}: power boundary and watts are inconsistent",
    )

    summary = result.get("summary")
    _require(isinstance(summary, dict), f"{source}: summary is invalid")
    _require(summary.get("slo") == slo, f"{source}: summary SLO mismatch")
    curve = summary.get("curve")
    _require(
        isinstance(curve, list) and len(curve) == len(agents),
        f"{source}: summary curve does not match the agent widths",
    )
    _require(
        [point.get("agents") for point in curve] == agents,
        f"{source}: summary curve is not in workload order",
    )

    runs = result.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == measured_runs * len(agents),
        f"{source}: run count does not match the workload",
    )
    tasks_by_width: dict[int, list[dict[str, Any]]] = {width: [] for width in agents}
    run_summaries_by_width: dict[int, list[dict[str, Any]]] = {
        width: [] for width in agents
    }
    seen: set[tuple[int, int]] = set()
    all_calls: list[dict[str, Any]] = []
    for run in runs:
        _require(isinstance(run, dict), f"{source}: run is not an object")
        width = run.get("agents")
        index = run.get("index")
        _require(width in agents, f"{source}: run agent width is not in the workload")
        _require(
            _is_int(index) and 0 <= index < measured_runs,
            f"{source}: run index is out of range",
        )
        key = (width, index)
        _require(
            key not in seen, f"{source}: duplicate run agents {width} index {index}"
        )
        seen.add(key)
        tasks = run.get("tasks")
        _require(
            isinstance(tasks, list) and len(tasks) == width,
            f"{source}: run {width} index {index} has the wrong task count",
        )
        _require(
            {task.get("agent_index") for task in tasks} == set(range(width)),
            f"{source}: run {width} index {index} did not cover every agent",
        )
        for task in tasks:
            _validate_agentic_task(
                task,
                trace,
                source,
                width,
                tool_wait_scale,
                deadline_slo=deadline_slo,
            )
            tasks_by_width[width].append(task)
            all_calls.extend(task["calls"])
        recomputed = summarize_agentic_run(
            width,
            tasks,
            slo,
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        )
        _require(
            _summary_value_matches(recomputed, run.get("summary")),
            f"{source}: run {width} index {index} summary does not match its tasks",
        )
        run_summaries_by_width[width].append(recomputed)

    for point in curve:
        _require(isinstance(point, dict), f"{source}: curve point is not an object")
        width = point.get("agents")
        _require(width in agents, f"{source}: curve agent width is not in the workload")
        recomputed = summarize_agentic_width(
            width,
            tasks_by_width[width],
            run_summaries_by_width[width],
            slo,
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        )
        if watts is not None and recomputed["meets_all_slo"]:
            recomputed["agents_at_slo_per_watt"] = width / watts
        _require(
            set(point) == set(recomputed),
            f"{source}: curve point keys do not match its runs",
        )
        for metric, expected_value in recomputed.items():
            _require(
                _summary_value_matches(expected_value, point[metric]),
                f"{source}: curve agent {width} metric {metric} does not match its runs",
            )

    if "population_outcomes" in summary or "stream_deadlines" in summary:
        total_span = sum(
            run["summary"].get("span_seconds", 0.0) for run in runs
        )
        expected_overall = summarize_population_and_deadlines(
            all_calls,
            duration_seconds=total_span if total_span > 0 else None,
            arrival_mode="closed_loop",
            offered_load={
                "agents": agents,
                "measured_runs": measured_runs,
                "calls_per_task": len(trace["calls"]),
                "offered_requests": len(all_calls),
            },
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
            legacy_slo_met=any(point["meets_all_slo"] for point in curve),
        )
        for key in ("population_outcomes", "stream_deadlines", "run_status"):
            _require(
                _summary_value_matches(expected_overall[key], summary.get(key)),
                f"{source}: summary {key} does not match its runs",
            )

    meeting = [point["agents"] for point in curve if point["meets_all_slo"]]
    expected_max = max(meeting) if meeting else None
    _require(
        summary.get("max_agents_meeting_all_slo") == expected_max,
        f"{source}: maximum agents at SLO does not match the curve",
    )
    throughput_points = [
        point for point in curve if point["tasks_per_second"] is not None
    ]
    expected_saturated = (
        max(throughput_points, key=lambda point: point["tasks_per_second"])["agents"]
        if throughput_points
        else None
    )
    _require(
        summary.get("saturated_agents") == expected_saturated,
        f"{source}: saturated agents do not match the curve",
    )

    slo_point = (
        next((point for point in curve if point["agents"] == expected_max), None)
        if expected_max is not None
        else None
    )
    expected_power = {
        "boundary": boundary,
        "watts": watts,
        "method": power.get("method"),
        "agents_at_slo_per_watt": (
            expected_max / watts
            if expected_max is not None and watts is not None
            else None
        ),
        "output_tokens_per_second_per_watt": (
            slo_point["output_tokens_per_second"] / watts
            if slo_point is not None
            and watts is not None
            and slo_point["output_tokens_per_second"] is not None
            else None
        ),
    }
    _require(
        set(power) == set(expected_power),
        f"{source}: power condition keys are invalid",
    )
    for key, expected_value in expected_power.items():
        _require(
            _summary_value_matches(expected_value, power.get(key)),
            f"{source}: power condition {key} does not match the curve",
        )
    return result


def _validate_agentic_task(
    task: Any,
    trace: dict[str, Any],
    source: str,
    agents: int,
    tool_wait_scale: float,
    *,
    deadline_slo: dict[str, float | None] | None = None,
) -> None:
    _require(isinstance(task, dict), f"{source}: task is not an object")
    agent_index = task.get("agent_index")
    _require(
        _is_int(agent_index) and 0 <= agent_index < agents,
        f"{source}: task agent_index is invalid",
    )
    call_count = len(trace["calls"])
    _require(
        task.get("call_count") == call_count,
        f"{source}: task call_count does not match the trace",
    )
    task_seconds = task.get("task_seconds")
    _require(
        _is_finite_number(task_seconds) and task_seconds >= 0,
        f"{source}: task_seconds is invalid",
    )
    start_offset = task.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: task start_offset_seconds is invalid",
    )
    scaled_waits = [
        call["tool_wait_seconds"] * tool_wait_scale for call in trace["calls"]
    ]
    _require(
        _summary_value_matches(sum(scaled_waits), task.get("tool_wait_seconds")),
        f"{source}: task tool_wait_seconds does not match the trace",
    )
    calls = task.get("calls")
    _require(
        isinstance(calls, list) and len(calls) == call_count,
        f"{source}: task calls do not match the trace",
    )
    previous_serialized = ""
    for index, call in enumerate(calls):
        _require(isinstance(call, dict), f"{source}: call is not an object")
        _require(
            call.get("agent_index") == agent_index,
            f"{source}: call agent_index does not match its task",
        )
        _require(
            call.get("call_index") == index,
            f"{source}: call_index does not match its position",
        )
        _require(
            call.get("max_tokens") == trace["calls"][index]["max_tokens"],
            f"{source}: call max_tokens does not match the trace",
        )
        _require(
            _summary_value_matches(
                scaled_waits[index], call.get("tool_wait_after_seconds")
            ),
            f"{source}: call tool wait does not match the trace",
        )
        serialized = serialize_messages(agentic_messages(trace, index))
        expected_reused = common_prefix_length(serialized, previous_serialized)
        previous_serialized = serialized
        _require(
            call.get("prompt_characters") == len(serialized),
            f"{source}: call prompt_characters does not match the trace",
        )
        _require(
            call.get("reused_prefix_characters") == expected_reused,
            f"{source}: call reused_prefix_characters does not match the trace",
        )
        _require(
            call.get("messages_sha256")
            == hashlib.sha256(serialized.encode()).hexdigest(),
            f"{source}: call messages_sha256 does not match the trace",
        )
        _validate_agentic_call(call, source, deadline_slo=deadline_slo)
        _require(
            call["start_offset_seconds"] <= task_seconds + DEADLINE_EPS,
            f"{source}: call start offset exceeds its task time",
        )
    if all(call.get("outcome", "completed") == "completed" for call in calls):
        _require(task_seconds > 0, f"{source}: completed task_seconds must be positive")
    _require(
        _summary_value_matches(
            _total_or_none(calls, "prompt_tokens"), task.get("prompt_tokens")
        ),
        f"{source}: task prompt_tokens does not match its calls",
    )
    for key in (
        "completion_tokens",
        "reused_prefix_characters",
        "reused_prefix_tokens_estimate",
        "new_input_tokens_estimate",
        "server_reported_cached_tokens",
    ):
        _require(
            _summary_value_matches(_total_or_none(calls, key), task.get(key)),
            f"{source}: task {key} does not match its calls",
        )


def _validate_agentic_call(
    call: dict[str, Any],
    source: str,
    *,
    deadline_slo: dict[str, float | None] | None = None,
) -> None:
    outcome = call.get("outcome", "completed")
    _require(
        outcome in REQUEST_OUTCOMES,
        f"{source}: call outcome {outcome!r} is invalid",
    )
    prompt_characters = call.get("prompt_characters")
    _require(
        _is_int(prompt_characters) and prompt_characters > 0,
        f"{source}: call prompt_characters is invalid",
    )
    reused = call.get("reused_prefix_characters")
    _require(
        _is_int(reused) and 0 <= reused <= prompt_characters,
        f"{source}: call reused_prefix_characters is invalid",
    )
    fraction = call.get("reused_prefix_fraction")
    _require(
        _is_finite_number(fraction)
        and math.isclose(
            fraction, reused / prompt_characters, rel_tol=1e-9, abs_tol=1e-9
        ),
        f"{source}: call reused_prefix_fraction is invalid",
    )
    prompt_tokens = call.get("prompt_tokens")
    _require(
        prompt_tokens is None or (_is_int(prompt_tokens) and prompt_tokens >= 0),
        f"{source}: call prompt_tokens is invalid",
    )
    reused_estimate = call.get("reused_prefix_tokens_estimate")
    new_estimate = call.get("new_input_tokens_estimate")
    if prompt_tokens is None:
        _require(
            reused_estimate is None and new_estimate is None,
            f"{source}: call token estimates exist without prompt_tokens",
        )
    else:
        _require(
            _is_int(reused_estimate)
            and _is_int(new_estimate)
            and reused_estimate + new_estimate == prompt_tokens
            and reused_estimate == round(prompt_tokens * reused / prompt_characters),
            f"{source}: call token estimates are invalid",
        )
    cached = call.get("server_reported_cached_tokens")
    _require(
        cached is None or (_is_int(cached) and cached >= 0),
        f"{source}: call server_reported_cached_tokens is invalid",
    )
    cached_source = call.get("server_reported_cached_tokens_source")
    if cached is None:
        _require(
            cached_source == "unreported",
            f"{source}: call cached token source is invalid",
        )
    else:
        _require(
            cached_source == "usage.prompt_tokens_details.cached_tokens",
            f"{source}: call cached token source is invalid",
        )
    _require(
        call.get("cache_confirmation") == "client-prefix-overlap-only",
        f"{source}: call cache confirmation is invalid",
    )
    start_offset = call.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: call start_offset_seconds is invalid",
    )

    if "stream_observation" in call:
        _validate_stream_observation(call, source, deadline_slo=deadline_slo)

    if outcome != "completed":
        _require(
            isinstance(call.get("outcome_reason"), str) and bool(call["outcome_reason"]),
            f"{source}: non-completed call must record outcome_reason",
        )
        return

    completion_tokens = call.get("completion_tokens")
    _require(
        _is_int(completion_tokens) and completion_tokens > 0,
        f"{source}: call completion_tokens is invalid",
    )
    ttft = call.get("ttft_seconds")
    last = call.get("last_token_seconds")
    wall = call.get("wall_seconds")
    _require(
        _is_finite_number(wall) and wall > 0,
        f"{source}: call wall_seconds is invalid",
    )
    for name in ("ttft_seconds", "last_token_seconds"):
        value = call.get(name)
        _require(
            _is_finite_number(value) and 0 <= value <= wall,
            f"{source}: call {name} is invalid",
        )
    _require(last >= ttft, f"{source}: call token times are inverted")
    expected_itl = (
        (last - ttft) / (completion_tokens - 1)
        if last > ttft and completion_tokens > 1
        else None
    )
    _require(
        _summary_value_matches(expected_itl, call.get("itl_seconds_mean")),
        f"{source}: call itl_seconds_mean does not match its timings",
    )
    expected_decode = expected_itl and 1 / expected_itl
    _require(
        _summary_value_matches(expected_decode, call.get("decode_tokens_per_second")),
        f"{source}: call decode_tokens_per_second does not match its timings",
    )
    _require(
        call.get("token_count_source") == "usage",
        f"{source}: call token_count_source is not usage",
    )
    text_hash = call.get("text_sha256")
    _require(
        isinstance(text_hash, str)
        and len(text_hash) == 64
        and all(character in "0123456789abcdef" for character in text_hash),
        f"{source}: call text_sha256 is not a SHA-256",
    )
    _require(
        _is_int(call.get("text_characters")) and call["text_characters"] > 0,
        f"{source}: call text_characters is invalid",
    )
    _require(
        _is_int(call.get("stream_chunks")) and call["stream_chunks"] >= 0,
        f"{source}: call stream_chunks is invalid",
    )
    _require(
        isinstance(call.get("finish_reason_seen"), bool),
        f"{source}: call finish_reason_seen is invalid",
    )


def load_agentic_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"{path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} is not JSON: {error}") from error
    return validate_agentic_result(result, str(path))


def parse_agent_widths(value: str) -> list[int]:
    widths = []
    for item in value.split(","):
        try:
            width = int(item)
        except ValueError as error:
            raise BenchmarkError(f"invalid agent width {item!r}") from error
        if width < 1:
            raise BenchmarkError("agent widths must be at least one")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise BenchmarkError("at least one agent width is required")
    return widths


def load_corpus(path: Path) -> dict[str, Any]:
    try:
        corpus = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read corpus {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"corpus {path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"corpus {path} is not JSON: {error}") from error
    if not isinstance(corpus, dict):
        raise BenchmarkError(f"corpus {path} must be an object")
    prompts = corpus.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise BenchmarkError(f"corpus {path} must contain non-empty prompts")
    return corpus


def redact_path(value: str) -> str:
    if value.startswith("/") and not value.startswith("/nix/store/"):
        return "<absolute-path>"
    return value


def selected_environment() -> dict[str, str]:
    selected = {}
    for key in ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        selected[key] = redact_path(value)
    return selected


def source_revision(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    configured = os.environ.get("QWEN38_BENCH_SOURCE_REVISION")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_benchmark(
    endpoint: Endpoint,
    corpus: dict[str, Any],
    max_tokens: int,
    concurrency: list[int],
    runs: int,
    warmup_runs: int,
    stream: bool,
    settle_seconds: float,
    revision: str | None,
    note: str | None,
    *,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    deadline_slo: dict[str, float | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if not _is_int(max_trace_events) or max_trace_events <= 0:
        raise BenchmarkError("max_trace_events must be a positive integer")
    norm_deadline_slo = _validate_deadline_slo(deadline_slo)
    if (
        not _is_int(max_tokens)
        or not _is_int(runs)
        or not _is_int(warmup_runs)
        or max_tokens <= 0
        or runs <= 0
        or warmup_runs < 0
    ):
        raise BenchmarkError(
            "max_tokens and runs must be positive integers; warmup_runs non-negative"
        )
    if (
        not concurrency
        or any(not _is_int(width) or width <= 1 for width in concurrency)
        or len(set(concurrency)) != len(concurrency)
    ):
        raise BenchmarkError(
            "concurrency widths must be unique integers greater than one"
        )
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise BenchmarkError("settle seconds must be finite and non-negative")
    prompts = corpus.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise BenchmarkError("corpus prompts are invalid")
    corpus_id = corpus.get("id", "unnamed")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise BenchmarkError("corpus id must be a non-empty string")
    model, probe = resolve_model(endpoint)
    workload = {
        "corpus_id": corpus_id,
        "corpus_chunk_sha256": canonical_sha256(prompts),
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "settle_seconds": settle_seconds,
        "concurrency": concurrency,
        "modes": ["single"] + [f"batch-{width}" for width in concurrency],
    }
    workload_sha256 = canonical_sha256(workload)

    started = utc_now()
    started_monotonic = clock()
    for _ in range(warmup_runs):
        run_single(
            endpoint,
            model,
            prompts,
            max_tokens,
            stream,
            clock=clock,
            max_trace_events=max_trace_events,
            deadline_slo=norm_deadline_slo,
        )
        for width in concurrency:
            run_batch_with_offsets(
                endpoint,
                model,
                prompts,
                width,
                max_tokens,
                stream,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=norm_deadline_slo,
            )
    if warmup_runs:
        time.sleep(settle_seconds)

    measured: list[dict[str, Any]] = []
    run_summaries: dict[str, list[dict[str, Any]]] = {
        "single": [],
        **{f"batch-{width}": [] for width in concurrency},
    }
    for run_index in range(runs):
        records = run_single(
            endpoint,
            model,
            prompts,
            max_tokens,
            stream,
            clock=clock,
            max_trace_events=max_trace_events,
            deadline_slo=norm_deadline_slo,
        )
        mode = "single"
        summary = summarize_run(mode, records)
        measured.append({
            "mode": mode,
            "index": run_index,
            "summary": summary,
            "records": records,
        })
        run_summaries[mode].append(summary)
        time.sleep(settle_seconds)
        for width in concurrency:
            mode = f"batch-{width}"
            records = run_batch_with_offsets(
                endpoint,
                model,
                prompts,
                width,
                max_tokens,
                stream,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=norm_deadline_slo,
            )
            summary = summarize_run(mode, records)
            measured.append({
                "mode": mode,
                "index": run_index,
                "summary": summary,
                "records": records,
            })
            run_summaries[mode].append(summary)
            time.sleep(settle_seconds)

    summaries = {
        mode: summarize_records(
            mode,
            [
                record
                for run in measured
                if run["mode"] == mode
                for record in run["records"]
            ],
            run_summaries[mode],
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        )
        for mode in run_summaries
    }
    finished = utc_now()
    return {
        "schema": SCHEMA,
        "schema_extensions": [STREAM_DEADLINE_SCHEMA],
        "generated_utc": finished,
        "started_utc": started,
        "duration_seconds": clock() - started_monotonic,
        "note": note,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "conditions": {
            "deadline_slo": norm_deadline_slo,
            "max_trace_events": max_trace_events,
            "arrival_mode": "closed_loop",
        },
        "target": {
            "label": endpoint.label,
            "base_url": endpoint.base_url,
            "model": model,
            "probe": probe,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": source_revision(revision),
            "selected_environment": selected_environment(),
        },
        "summary": summaries,
        "runs": measured,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _summary_value_matches(expected: Any, stored: Any) -> bool:
    if expected is None or stored is None:
        return expected is stored
    if isinstance(expected, bool) or isinstance(stored, bool):
        return type(expected) is type(stored) and expected == stored
    if isinstance(expected, (int, float)):
        return _is_finite_number(stored) and math.isclose(
            stored, expected, rel_tol=1e-9, abs_tol=1e-9
        )
    if isinstance(expected, dict):
        return (
            isinstance(stored, dict)
            and set(expected) == set(stored)
            and all(_summary_value_matches(expected[key], stored[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(stored, list)
            and len(expected) == len(stored)
            and all(
                _summary_value_matches(left, right)
                for left, right in zip(expected, stored, strict=True)
            )
        )
    return stored == expected


def _validate_record(
    record: Any,
    source: str,
    prompts: list[str],
    *,
    deadline_slo: dict[str, float | None] | None = None,
) -> None:
    _require(isinstance(record, dict), f"{source}: run record is not an object")
    prompt_index = record.get("prompt_index")
    _require(
        _is_int(prompt_index) and 0 <= prompt_index < len(prompts),
        f"{source}: record prompt_index is out of range",
    )
    prompt_hash = hashlib.sha256(prompts[prompt_index].encode()).hexdigest()
    _require(
        record.get("prompt_sha256") == prompt_hash,
        f"{source}: record prompt_sha256 does not match the corpus",
    )
    text_hash = record.get("text_sha256")
    _require(
        isinstance(text_hash, str)
        and len(text_hash) == 64
        and all(character in "0123456789abcdef" for character in text_hash),
        f"{source}: record text_sha256 is not a SHA-256",
    )
    completion_tokens = record.get("completion_tokens")
    _require(
        _is_int(completion_tokens) and completion_tokens > 0,
        f"{source}: record completion_tokens is invalid",
    )
    text_characters = record.get("text_characters")
    _require(
        _is_int(text_characters) and text_characters > 0,
        f"{source}: record text_characters is invalid",
    )
    start_offset = record.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: record start_offset_seconds is invalid",
    )
    wall = record.get("wall_seconds")
    _require(
        _is_finite_number(wall) and wall >= 0,
        f"{source}: record wall_seconds is invalid",
    )
    if completion_tokens:
        _require(wall > 0, f"{source}: record wall_seconds is not positive")
    for name in ("ttft_seconds", "last_token_seconds"):
        value = record.get(name)
        if value is not None:
            _require(
                _is_finite_number(value) and value >= 0 and value <= wall,
                f"{source}: record {name} is invalid",
            )
    ttft = record.get("ttft_seconds")
    last_token = record.get("last_token_seconds")
    if ttft is not None and last_token is not None:
        _require(last_token >= ttft, f"{source}: record token times are inverted")

    expected_decode = None
    if (
        ttft is not None
        and last_token is not None
        and last_token > ttft
        and completion_tokens > 1
    ):
        expected_decode = (completion_tokens - 1) / (last_token - ttft)
    _require(
        _summary_value_matches(expected_decode, record.get("decode_tokens_per_second")),
        f"{source}: record decode_tokens_per_second does not match its timings",
    )
    expected_e2e = completion_tokens / wall if wall > 0 else None
    _require(
        _summary_value_matches(expected_e2e, record.get("e2e_tokens_per_second")),
        f"{source}: record e2e_tokens_per_second does not match its timings",
    )

    prompt_tokens = record.get("prompt_tokens")
    _require(
        prompt_tokens is None or (_is_int(prompt_tokens) and prompt_tokens >= 0),
        f"{source}: record prompt_tokens is invalid",
    )
    stream_chunks = record.get("stream_chunks")
    _require(
        _is_int(stream_chunks) and stream_chunks >= 0,
        f"{source}: record stream_chunks is invalid",
    )
    _require(
        record.get("token_count_source") == "usage",
        f"{source}: record token_count_source is not usage",
    )
    if "finish_reason_seen" in record:
        _require(
            isinstance(record["finish_reason_seen"], bool),
            f"{source}: record finish_reason_seen is invalid",
        )
    if "stream_observation" in record:
        _validate_stream_observation(record, source, deadline_slo=deadline_slo)


def validate_result(result: Any, source: str = "result") -> dict[str, Any]:
    _require(isinstance(result, dict), f"{source}: result is not an object")
    _require(result.get("schema") == SCHEMA, f"{source}: schema is not {SCHEMA}")
    target = result.get("target")
    _require(
        isinstance(target, dict)
        and isinstance(target.get("label"), str)
        and bool(target["label"]),
        f"{source}: target label is missing",
    )
    _require(
        isinstance(target.get("base_url"), str) and bool(target["base_url"]),
        f"{source}: target base_url is missing",
    )
    _require(
        isinstance(target.get("model"), str) and bool(target["model"]),
        f"{source}: target model is missing",
    )
    _require(
        isinstance(result.get("environment"), dict),
        f"{source}: environment is not an object",
    )
    workload = result.get("workload")
    _require(isinstance(workload, dict), f"{source}: workload is not an object")
    digest = result.get("workload_sha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and canonical_sha256(workload) == digest,
        f"{source}: workload digest does not match its workload",
    )
    _require(
        isinstance(workload.get("corpus_id"), str)
        and bool(workload["corpus_id"].strip()),
        f"{source}: workload corpus id is invalid",
    )
    prompts = workload.get("prompts")
    _require(
        isinstance(prompts, list)
        and bool(prompts)
        and all(isinstance(prompt, str) and bool(prompt.strip()) for prompt in prompts),
        f"{source}: workload prompts are invalid",
    )
    measured_runs = workload.get("measured_runs")
    _require(
        _is_int(measured_runs) and measured_runs > 0,
        f"{source}: workload measured_runs is invalid",
    )
    warmup_runs = workload.get("warmup_runs")
    _require(
        _is_int(warmup_runs) and warmup_runs >= 0,
        f"{source}: workload warmup_runs is invalid",
    )
    _require(
        _is_int(workload.get("max_tokens")) and workload["max_tokens"] > 0,
        f"{source}: workload max_tokens is invalid",
    )
    _require(
        workload.get("corpus_chunk_sha256") == canonical_sha256(prompts),
        f"{source}: workload corpus digest does not match its prompts",
    )
    _require(
        workload.get("temperature") == 0,
        f"{source}: workload temperature is not greedy",
    )
    _require(
        isinstance(workload.get("stream"), bool),
        f"{source}: workload stream flag is invalid",
    )
    _require(
        _is_finite_number(workload.get("settle_seconds"))
        and workload["settle_seconds"] >= 0,
        f"{source}: workload settle_seconds is invalid",
    )
    concurrency = workload.get("concurrency")
    _require(
        isinstance(concurrency, list)
        and bool(concurrency)
        and all(_is_int(width) and width > 1 for width in concurrency)
        and len(set(concurrency)) == len(concurrency),
        f"{source}: workload concurrency is invalid",
    )
    expected_modes = ["single"] + [f"batch-{width}" for width in concurrency]
    _require(
        workload.get("modes") == expected_modes,
        f"{source}: workload modes do not match its concurrency",
    )

    conditions = result.get("conditions")
    deadline_slo = None
    if isinstance(conditions, dict) and "deadline_slo" in conditions:
        deadline_slo = _validate_deadline_slo(conditions.get("deadline_slo"))

    summary = result.get("summary")
    _require(
        isinstance(summary, dict) and set(summary) == set(expected_modes),
        f"{source}: summary modes do not match the workload",
    )
    runs = result.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == measured_runs * len(expected_modes),
        f"{source}: run count does not match the workload",
    )
    records_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in expected_modes
    }
    run_summaries: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in expected_modes
    }
    seen: set[tuple[str, int]] = set()
    for run in runs:
        _require(isinstance(run, dict), f"{source}: run is not an object")
        mode = run.get("mode")
        index = run.get("index")
        _require(
            mode in expected_modes,
            f"{source}: run mode is not in the workload",
        )
        _require(
            _is_int(index) and 0 <= index < measured_runs,
            f"{source}: run index is out of range",
        )
        key = (mode, index)
        _require(key not in seen, f"{source}: duplicate run {mode} index {index}")
        seen.add(key)
        expected_records = (
            len(prompts) if mode == "single" else int(mode.removeprefix("batch-"))
        )
        records = run.get("records")
        _require(
            isinstance(records, list) and len(records) == expected_records,
            f"{source}: run {mode} index {index} has the wrong record count",
        )
        for record in records:
            _validate_record(record, source, prompts, deadline_slo=deadline_slo)
            records_by_mode[mode].append(record)
        if mode == "single":
            _require(
                len({record["prompt_index"] for record in records}) == len(prompts),
                f"{source}: a single-stream run did not cover every prompt",
            )
        run_summaries[mode].append(summarize_run(mode, records))

    for mode in expected_modes:
        recomputed = summarize_records(
            mode,
            records_by_mode[mode],
            run_summaries[mode],
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        )
        stored = summary[mode]
        _require(
            isinstance(stored, dict) and set(stored) == set(recomputed),
            f"{source}: {mode} summary keys do not match its runs",
        )
        for metric, expected_value in recomputed.items():
            _require(
                _summary_value_matches(expected_value, stored[metric]),
                f"{source}: {mode} summary metric {metric} does not match its runs",
            )
    return result


def load_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"{path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} is not JSON: {error}") from error
    return validate_result(result, str(path))


def validate_interference_workload(workload: Any, source: str) -> dict[str, Any]:
    _require(isinstance(workload, dict), f"{source}: interference workload is not an object")
    _require(
        workload.get("schema_version") == 1,
        f"{source}: interference workload schema_version is not 1",
    )
    workload_id = workload.get("id")
    _require(
        isinstance(workload_id, str) and bool(workload_id.strip()),
        f"{source}: interference workload id is invalid",
    )
    _require(
        _is_int(workload.get("seed")),
        f"{source}: interference workload seed is invalid",
    )
    system_prompt = workload.get("system_prompt")
    _require(
        isinstance(system_prompt, str) and bool(system_prompt.strip()),
        f"{source}: interference workload system_prompt is invalid",
    )
    victims = workload.get("victim_streams")
    _require(
        isinstance(victims, list) and bool(victims),
        f"{source}: interference workload must contain victim_streams",
    )
    victim_ids: set[str] = set()
    for index, item in enumerate(victims):
        _require(isinstance(item, dict), f"{source}: victim {index} is not an object")
        stream_id = item.get("stream_id")
        _require(
            isinstance(stream_id, str)
            and bool(stream_id.strip())
            and stream_id not in victim_ids,
            f"{source}: victim {index} stream_id is invalid or duplicate",
        )
        victim_ids.add(stream_id)
        _require(
            _is_finite_number(item.get("dispatch_offset_seconds"))
            and item["dispatch_offset_seconds"] >= 0,
            f"{source}: victim {stream_id} dispatch_offset_seconds is invalid",
        )
        _require(
            _is_int(item.get("max_tokens")) and item["max_tokens"] > 0,
            f"{source}: victim {stream_id} max_tokens is invalid",
        )
        prompt = item.get("prompt")
        _require(
            isinstance(prompt, str) and bool(prompt.strip()),
            f"{source}: victim {stream_id} prompt is invalid",
        )

    schedule = workload.get("burst_schedule")
    _require(isinstance(schedule, dict), f"{source}: burst_schedule is not an object")
    _require(
        _is_finite_number(schedule.get("burst_offset_seconds"))
        and schedule["burst_offset_seconds"] >= 0,
        f"{source}: burst_offset_seconds is invalid",
    )
    _require(
        _is_finite_number(schedule.get("jitter_window_seconds"))
        and schedule["jitter_window_seconds"] >= 0,
        f"{source}: jitter_window_seconds is invalid",
    )
    for key in ("long_prompts", "short_control_prompts"):
        items = schedule.get(key)
        _require(
            isinstance(items, list) and bool(items),
            f"{source}: burst_schedule.{key} must be a non-empty list",
        )
        burst_ids: set[str] = set()
        for index, item in enumerate(items):
            _require(isinstance(item, dict), f"{source}: {key}[{index}] is not an object")
            burst_id = item.get("burst_id")
            _require(
                isinstance(burst_id, str)
                and bool(burst_id.strip())
                and burst_id not in burst_ids,
                f"{source}: {key}[{index}] burst_id is invalid or duplicate",
            )
            burst_ids.add(burst_id)
            _require(
                _is_finite_number(item.get("relative_offset_seconds"))
                and item["relative_offset_seconds"] >= 0,
                f"{source}: {burst_id} relative_offset_seconds is invalid",
            )
            _require(
                _is_int(item.get("max_tokens")) and item["max_tokens"] > 0,
                f"{source}: {burst_id} max_tokens is invalid",
            )
            prompt = item.get("prompt")
            _require(
                isinstance(prompt, str) and bool(prompt.strip()),
                f"{source}: {burst_id} prompt is invalid",
            )
    _require(
        len(schedule["long_prompts"]) == len(schedule["short_control_prompts"]),
        f"{source}: long_prompts and short_control_prompts must have equal length",
    )
    return workload


def load_interference_workload(path: Path) -> dict[str, Any]:
    try:
        workload = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read interference workload {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(
            f"interference workload {path} is not UTF-8: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            f"interference workload {path} is not JSON: {error}"
        ) from error
    return validate_interference_workload(workload, str(path))


def interference_workload_manifest(
    fixture: dict[str, Any],
    *,
    control_mode: str,
    schedule_scale: float,
    runs: int,
    warmup_runs: int,
    settle_seconds: float,
    cache_mode: str,
) -> dict[str, Any]:
    burst_offset = fixture["burst_schedule"]["burst_offset_seconds"]
    return {
        "fixture_id": fixture["id"],
        "fixture_sha256": canonical_sha256(fixture),
        "seed": fixture["seed"],
        "system_prompt_sha256": hashlib.sha256(
            fixture["system_prompt"].encode()
        ).hexdigest(),
        "control_mode": control_mode,
        "schedule_scale": schedule_scale,
        "measured_runs": runs,
        "warmup_runs": warmup_runs,
        "settle_seconds": settle_seconds,
        "temperature": 0,
        "stream": True,
        "cache_mode": cache_mode,
        "victim_streams": [
            {
                "stream_id": item["stream_id"],
                "planned_dispatch_offset_seconds": (
                    item["dispatch_offset_seconds"] * schedule_scale
                ),
                "max_tokens": item["max_tokens"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                "prompt_characters": len(item["prompt"]),
            }
            for item in fixture["victim_streams"]
        ],
        "interference_burst": [
            {
                "burst_id": item["burst_id"],
                "planned_dispatch_offset_seconds": (
                    (burst_offset + item["relative_offset_seconds"]) * schedule_scale
                ),
                "max_tokens": item["max_tokens"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                "prompt_characters": len(item["prompt"]),
            }
            for item in fixture["burst_schedule"]["long_prompts"]
        ],
        "control_burst": (
            [
                {
                    "burst_id": item["burst_id"],
                    "planned_dispatch_offset_seconds": (
                        (burst_offset + item["relative_offset_seconds"])
                        * schedule_scale
                    ),
                    "max_tokens": item["max_tokens"],
                    "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                    "prompt_characters": len(item["prompt"]),
                }
                for item in fixture["burst_schedule"]["short_control_prompts"]
            ]
            if control_mode == "short_prompt"
            else []
        ),
    }


def _summarize_stream_group(
    records: list[dict[str, Any]],
    *,
    duration_seconds: float | None,
    arrival_mode: str,
    offered_load: dict[str, Any],
    warmup_runs: int,
    deadline_slo: dict[str, float | None] | None,
) -> dict[str, Any]:
    completed = [r for r in records if r.get("outcome", "completed") == "completed"]
    ttfts = [r["ttft_seconds"] for r in completed if r.get("ttft_seconds") is not None]
    walls = [r["wall_seconds"] for r in completed if r.get("wall_seconds") is not None]
    itls = [
        (r["last_token_seconds"] - r["ttft_seconds"]) / (r["completion_tokens"] - 1)
        for r in completed
        if r.get("ttft_seconds") is not None
        and r.get("last_token_seconds") is not None
        and r["last_token_seconds"] > r["ttft_seconds"]
        and _is_int(r.get("completion_tokens"))
        and r["completion_tokens"] > 1
    ]
    ext = summarize_population_and_deadlines(
        records,
        duration_seconds=duration_seconds,
        arrival_mode=arrival_mode,
        offered_load=offered_load,
        warmup_runs=warmup_runs,
        deadline_slo=deadline_slo,
    )
    return {
        "requests": len(records),
        "ttft_seconds_median": median(ttfts),
        "ttft_seconds_p95": percentile(ttfts, 0.95),
        "ttft_seconds_max": max(ttfts) if ttfts else None,
        "itl_seconds_median": median(itls),
        "itl_seconds_p95": percentile(itls, 0.95),
        "task_latency_seconds_median": median(walls),
        "task_latency_seconds_p95": percentile(walls, 0.95),
        "task_latency_seconds_max": max(walls) if walls else None,
        "population_outcomes": ext["population_outcomes"],
        "stream_deadlines": ext["stream_deadlines"],
        "run_status": ext["run_status"],
    }


def _phase_span_seconds(streams: list[dict[str, Any]]) -> float | None:
    starts = [
        s["actual_dispatch_offset_seconds"]
        for s in streams
        if _is_finite_number(s.get("actual_dispatch_offset_seconds"))
    ]
    ends = [
        s["completion_offset_seconds"]
        for s in streams
        if _is_finite_number(s.get("completion_offset_seconds"))
    ]
    if not starts or not ends:
        return None
    span = max(ends) - min(starts)
    return span if span > 0 else None


def summarize_interference_phase(
    phase: str,
    streams: list[dict[str, Any]],
    *,
    control_mode: str,
    warmup_runs: int,
    deadline_slo: dict[str, float | None] | None,
) -> dict[str, Any]:
    span = _phase_span_seconds(streams)
    victims = [s for s in streams if s.get("role") == "victim"]
    burst = [s for s in streams if s.get("role") != "victim"]
    arrival_mode = (
        "interference_burst"
        if phase == "interference"
        else f"control_{control_mode}"
    )
    victim_summary = _summarize_stream_group(
        victims,
        duration_seconds=span,
        arrival_mode=arrival_mode,
        offered_load={
            "phase": phase,
            "role": "victim",
            "victim_streams": len(victims),
            "burst_requests": len(burst),
            "offered_requests": len(victims),
        },
        warmup_runs=warmup_runs,
        deadline_slo=deadline_slo,
    )
    all_summary = _summarize_stream_group(
        streams,
        duration_seconds=span,
        arrival_mode=arrival_mode,
        offered_load={
            "phase": phase,
            "role": "all",
            "victim_streams": len(victims),
            "burst_requests": len(burst),
            "offered_requests": len(streams),
        },
        warmup_runs=warmup_runs,
        deadline_slo=None,
    )
    burst_summary = (
        _summarize_stream_group(
            burst,
            duration_seconds=span,
            arrival_mode=arrival_mode,
            offered_load={
                "phase": phase,
                "role": "burst",
                "victim_streams": len(victims),
                "burst_requests": len(burst),
                "offered_requests": len(burst),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        )
        if burst
        else None
    )
    return {
        "phase": phase,
        "span_seconds": span,
        "victims": victim_summary,
        "burst": burst_summary,
        "all_requests": all_summary,
        "server_side_phase_attribution": dict(SERVER_PHASE_ATTRIBUTION_UNAVAILABLE),
    }


def _metric_delta_ratio(
    control_value: float | None, interference_value: float | None
) -> dict[str, float | None]:
    delta = None
    ratio = None
    if _is_finite_number(control_value) and _is_finite_number(interference_value):
        delta = float(interference_value) - float(control_value)
        if float(control_value) != 0.0:
            ratio = float(interference_value) / float(control_value)
    return {
        "control": control_value,
        "interference": interference_value,
        "delta": delta,
        "ratio": ratio,
    }


def summarize_interference_comparison(
    control_victims: dict[str, Any], interference_victims: dict[str, Any]
) -> dict[str, Any]:
    ctrl_ev = control_victims["stream_deadlines"]["event_weighted"]
    int_ev = interference_victims["stream_deadlines"]["event_weighted"]
    ctrl_req = control_victims["stream_deadlines"]["request_weighted"]
    int_req = interference_victims["stream_deadlines"]["request_weighted"]
    ctrl_mg = control_victims["stream_deadlines"]["max_gap_slo_compliance"]
    int_mg = interference_victims["stream_deadlines"]["max_gap_slo_compliance"]
    ctrl_td = control_victims["stream_deadlines"][
        "cumulative_token_deadline_compliance"
    ]
    int_td = interference_victims["stream_deadlines"][
        "cumulative_token_deadline_compliance"
    ]
    ctrl_pop = control_victims["population_outcomes"]
    int_pop = interference_victims["population_outcomes"]
    return {
        "ttft_seconds_p95": _metric_delta_ratio(
            control_victims["ttft_seconds_p95"],
            interference_victims["ttft_seconds_p95"],
        ),
        "task_latency_seconds_p95": _metric_delta_ratio(
            control_victims["task_latency_seconds_p95"],
            interference_victims["task_latency_seconds_p95"],
        ),
        "event_weighted_gap_seconds_p95": _metric_delta_ratio(
            ctrl_ev["event_gap_seconds_p95"], int_ev["event_gap_seconds_p95"]
        ),
        "event_weighted_gap_seconds_p99": _metric_delta_ratio(
            ctrl_ev["event_gap_seconds_p99"], int_ev["event_gap_seconds_p99"]
        ),
        "event_weighted_gap_seconds_max": _metric_delta_ratio(
            ctrl_ev["event_gap_seconds_max"], int_ev["event_gap_seconds_max"]
        ),
        "request_weighted_max_gap_seconds_p95": _metric_delta_ratio(
            ctrl_req["per_request_max_gap_seconds_p95"],
            int_req["per_request_max_gap_seconds_p95"],
        ),
        "request_weighted_max_gap_seconds_max": _metric_delta_ratio(
            ctrl_req["per_request_max_gap_seconds_max"],
            int_req["per_request_max_gap_seconds_max"],
        ),
        "max_gap_compliance_rate": _metric_delta_ratio(
            ctrl_mg["compliance_rate"], int_mg["compliance_rate"]
        ),
        "token_deadline_compliance_status": {
            "control": ctrl_td["status"],
            "interference": int_td["status"],
            "control_rate": ctrl_td["request_weighted_compliance_rate"],
            "interference_rate": int_td["request_weighted_compliance_rate"],
        },
        "failure_rate": _metric_delta_ratio(
            ctrl_pop["failure_rate"], int_pop["failure_rate"]
        ),
        "unknown_or_unobservable_rate": _metric_delta_ratio(
            ctrl_pop["unknown_or_unobservable_rate"],
            int_pop["unknown_or_unobservable_rate"],
        ),
        "server_side_phase_attribution": dict(SERVER_PHASE_ATTRIBUTION_UNAVAILABLE),
    }


def run_prefill_interference_phase(
    endpoint: Endpoint,
    model: str,
    fixture: dict[str, Any],
    *,
    phase: str,
    control_mode: str,
    schedule_scale: float,
    deadline_slo: dict[str, float | None] | None,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run one matched control or prefill-interference burst phase."""
    burst_offset = fixture["burst_schedule"]["burst_offset_seconds"]
    specs: list[dict[str, Any]] = []
    for item in fixture["victim_streams"]:
        specs.append({
            "stream_id": item["stream_id"],
            "role": "victim",
            "planned_offset": item["dispatch_offset_seconds"] * schedule_scale,
            "max_tokens": item["max_tokens"],
            "prompt": item["prompt"],
            "deadline_slo": deadline_slo,
        })
    if phase == "interference":
        for item in fixture["burst_schedule"]["long_prompts"]:
            specs.append({
                "stream_id": item["burst_id"],
                "role": "interference_burst",
                "planned_offset": (burst_offset + item["relative_offset_seconds"])
                * schedule_scale,
                "max_tokens": item["max_tokens"],
                "prompt": item["prompt"],
                "deadline_slo": None,
            })
    elif phase == "control" and control_mode == "short_prompt":
        for item in fixture["burst_schedule"]["short_control_prompts"]:
            specs.append({
                "stream_id": item["burst_id"],
                "role": "control_burst",
                "planned_offset": (burst_offset + item["relative_offset_seconds"])
                * schedule_scale,
                "max_tokens": item["max_tokens"],
                "prompt": item["prompt"],
                "deadline_slo": None,
            })

    barrier = threading.Barrier(len(specs))
    phase_start_box: list[float] = []
    phase_lock = threading.Lock()
    results: list[dict[str, Any] | None] = [None] * len(specs)
    system_prompt = fixture["system_prompt"]

    def worker(index: int, spec: dict[str, Any]) -> None:
        try:
            barrier.wait(timeout=endpoint.timeout)
        except BaseException as error:  # noqa: BLE001
            now = clock()
            results[index] = _failure_stream_record(
                outcome="incomplete",
                outcome_reason="barrier_aborted",
                error_message=f"{endpoint.label}: barrier aborted: {error}",
                start=now,
                end=now,
                max_trace_events=max_trace_events,
                deadline_slo=spec["deadline_slo"],
            )
            results[index].update({
                "stream_id": spec["stream_id"],
                "role": spec["role"],
                "max_tokens": spec["max_tokens"],
                "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
                "prompt_characters": len(spec["prompt"]),
                "planned_dispatch_offset_seconds": spec["planned_offset"],
                "actual_dispatch_offset_seconds": 0.0,
                "completion_offset_seconds": 0.0,
            })
            return

        with phase_lock:
            if not phase_start_box:
                phase_start_box.append(clock())
            phase_start = phase_start_box[0]

        delay = spec["planned_offset"] - (clock() - phase_start)
        if delay > 0:
            sleeper(delay)

        dispatch_time = clock()
        connection: http.client.HTTPConnection | None = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": spec["prompt"]},
        ]
        try:
            connection = open_connection(endpoint)
            dispatch_time = clock()
            record, _usage = observe_stream_completion(
                connection,
                endpoint,
                endpoint.api_path("/chat/completions"),
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": spec["max_tokens"],
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                f"{phase} {spec['stream_id']}",
                dispatch_time,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=spec["deadline_slo"],
            )
        except (TimeoutError, socket.timeout) as error:
            record = _failure_stream_record(
                outcome="timed_out",
                outcome_reason="connect_timeout",
                error_message=f"{endpoint.label}: connect timeout: {error}",
                start=dispatch_time,
                end=clock(),
                max_trace_events=max_trace_events,
                deadline_slo=spec["deadline_slo"],
            )
        except ConnectionRefusedError as error:
            record = _failure_stream_record(
                outcome="rejected",
                outcome_reason="connection_refused",
                error_message=f"{endpoint.label}: connection refused: {error}",
                start=dispatch_time,
                end=clock(),
                max_trace_events=max_trace_events,
                deadline_slo=spec["deadline_slo"],
            )
        except (OSError, http.client.HTTPException) as error:
            record = _failure_stream_record(
                outcome="incomplete",
                outcome_reason=f"connect_{type(error).__name__}",
                error_message=(
                    f"{endpoint.label}: connect error: {type(error).__name__}: {error}"
                ),
                start=dispatch_time,
                end=clock(),
                max_trace_events=max_trace_events,
                deadline_slo=spec["deadline_slo"],
            )
        finally:
            if connection is not None:
                connection.close()

        actual_offset = max(0.0, dispatch_time - phase_start)
        wall = record.get("wall_seconds")
        completion_offset = (
            actual_offset + wall
            if _is_finite_number(wall)
            else max(actual_offset, clock() - phase_start)
        )
        record.update({
            "stream_id": spec["stream_id"],
            "role": spec["role"],
            "max_tokens": spec["max_tokens"],
            "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
            "prompt_characters": len(spec["prompt"]),
            "planned_dispatch_offset_seconds": spec["planned_offset"],
            "actual_dispatch_offset_seconds": actual_offset,
            "completion_offset_seconds": completion_offset,
        })
        results[index] = record

    threads = [
        threading.Thread(
            target=worker,
            args=(index, spec),
            name=f"{phase}-{spec['stream_id']}",
        )
        for index, spec in enumerate(specs)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return [item for item in results if item is not None]


def run_prefill_interference_benchmark(
    endpoint: Endpoint,
    fixture: dict[str, Any],
    *,
    control_mode: str = "short_prompt",
    runs: int = 2,
    warmup_runs: int = 1,
    settle_seconds: float = 1.0,
    schedule_scale: float = 1.0,
    deadline_slo: dict[str, float | None],
    cache_mode: str = "unknown",
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    revision: str | None = None,
    note: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    validate_interference_workload(fixture, "interference workload")
    if control_mode not in CONTROL_MODES:
        raise BenchmarkError(
            f"control mode must be one of {', '.join(CONTROL_MODES)}"
        )
    if not _is_int(runs) or runs <= 0 or not _is_int(warmup_runs) or warmup_runs < 0:
        raise BenchmarkError("runs must be positive; warmup_runs non-negative")
    if not _is_finite_number(settle_seconds) or settle_seconds < 0:
        raise BenchmarkError("settle seconds must be finite and non-negative")
    if not _is_finite_number(schedule_scale) or schedule_scale < 0:
        raise BenchmarkError("schedule scale must be finite and non-negative")
    if cache_mode not in CACHE_MODES:
        raise BenchmarkError(f"cache mode must be one of {', '.join(CACHE_MODES)}")
    if not _is_int(max_trace_events) or max_trace_events <= 0:
        raise BenchmarkError("max_trace_events must be a positive integer")
    norm_deadline_slo = _validate_deadline_slo(deadline_slo)
    if (
        norm_deadline_slo["ttft_slo_seconds"] is None
        or norm_deadline_slo["tbt_slo_seconds"] is None
    ):
        raise BenchmarkError(
            "interference mode requires both ttft_slo_seconds and tbt_slo_seconds"
        )
    if norm_deadline_slo["max_gap_slo_seconds"] is None:
        norm_deadline_slo["max_gap_slo_seconds"] = norm_deadline_slo["tbt_slo_seconds"]

    workload = interference_workload_manifest(
        fixture,
        control_mode=control_mode,
        schedule_scale=schedule_scale,
        runs=runs,
        warmup_runs=warmup_runs,
        settle_seconds=settle_seconds,
        cache_mode=cache_mode,
    )
    workload_sha256 = canonical_sha256(workload)
    fixture_sha256 = workload["fixture_sha256"]
    model, probe = resolve_model(endpoint)

    started = utc_now()
    started_monotonic = clock()
    for _ in range(warmup_runs):
        run_prefill_interference_phase(
            endpoint,
            model,
            fixture,
            phase="control",
            control_mode=control_mode,
            schedule_scale=schedule_scale,
            deadline_slo=norm_deadline_slo,
            max_trace_events=max_trace_events,
            clock=clock,
            sleeper=sleeper,
        )
        run_prefill_interference_phase(
            endpoint,
            model,
            fixture,
            phase="interference",
            control_mode=control_mode,
            schedule_scale=schedule_scale,
            deadline_slo=norm_deadline_slo,
            max_trace_events=max_trace_events,
            clock=clock,
            sleeper=sleeper,
        )
    if warmup_runs and settle_seconds > 0:
        sleeper(settle_seconds)

    measured: list[dict[str, Any]] = []
    for run_index in range(runs):
        control_streams = run_prefill_interference_phase(
            endpoint,
            model,
            fixture,
            phase="control",
            control_mode=control_mode,
            schedule_scale=schedule_scale,
            deadline_slo=norm_deadline_slo,
            max_trace_events=max_trace_events,
            clock=clock,
            sleeper=sleeper,
        )
        if settle_seconds > 0:
            sleeper(settle_seconds)
        interference_streams = run_prefill_interference_phase(
            endpoint,
            model,
            fixture,
            phase="interference",
            control_mode=control_mode,
            schedule_scale=schedule_scale,
            deadline_slo=norm_deadline_slo,
            max_trace_events=max_trace_events,
            clock=clock,
            sleeper=sleeper,
        )
        if settle_seconds > 0:
            sleeper(settle_seconds)

        control_summary = summarize_interference_phase(
            "control",
            control_streams,
            control_mode=control_mode,
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        )
        interference_summary = summarize_interference_phase(
            "interference",
            interference_streams,
            control_mode=control_mode,
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        )
        measured.append({
            "index": run_index,
            "control": {
                "summary": control_summary,
                "streams": control_streams,
            },
            "interference": {
                "summary": interference_summary,
                "streams": interference_streams,
            },
            "comparison": summarize_interference_comparison(
                control_summary["victims"], interference_summary["victims"]
            ),
        })

    all_control_streams = [
        stream for run in measured for stream in run["control"]["streams"]
    ]
    all_interference_streams = [
        stream for run in measured for stream in run["interference"]["streams"]
    ]
    ctrl_spans = [
        run["control"]["summary"]["span_seconds"]
        for run in measured
        if run["control"]["summary"]["span_seconds"] is not None
    ]
    int_spans = [
        run["interference"]["summary"]["span_seconds"]
        for run in measured
        if run["interference"]["summary"]["span_seconds"] is not None
    ]
    ctrl_victims = [s for s in all_control_streams if s["role"] == "victim"]
    int_victims = [s for s in all_interference_streams if s["role"] == "victim"]
    ctrl_burst = [s for s in all_control_streams if s["role"] != "victim"]
    int_burst = [s for s in all_interference_streams if s["role"] != "victim"]

    control_overall = {
        "victims": _summarize_stream_group(
            ctrl_victims,
            duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
            arrival_mode=f"control_{control_mode}",
            offered_load={
                "phase": "control",
                "control_mode": control_mode,
                "measured_runs": runs,
                "victim_streams_per_run": len(fixture["victim_streams"]),
                "burst_requests_per_run": (
                    len(fixture["burst_schedule"]["short_control_prompts"])
                    if control_mode == "short_prompt"
                    else 0
                ),
                "offered_requests": len(ctrl_victims),
            },
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        ),
        "burst": (
            _summarize_stream_group(
                ctrl_burst,
                duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
                arrival_mode=f"control_{control_mode}",
                offered_load={
                    "phase": "control",
                    "role": "burst",
                    "measured_runs": runs,
                    "offered_requests": len(ctrl_burst),
                },
                warmup_runs=warmup_runs,
                deadline_slo=None,
            )
            if ctrl_burst
            else None
        ),
        "all_requests": _summarize_stream_group(
            all_control_streams,
            duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
            arrival_mode=f"control_{control_mode}",
            offered_load={
                "phase": "control",
                "control_mode": control_mode,
                "measured_runs": runs,
                "offered_requests": len(all_control_streams),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
    }
    interference_overall = {
        "victims": _summarize_stream_group(
            int_victims,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "measured_runs": runs,
                "victim_streams_per_run": len(fixture["victim_streams"]),
                "burst_requests_per_run": len(
                    fixture["burst_schedule"]["long_prompts"]
                ),
                "offered_requests": len(int_victims),
            },
            warmup_runs=warmup_runs,
            deadline_slo=norm_deadline_slo,
        ),
        "burst": _summarize_stream_group(
            int_burst,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "role": "burst",
                "measured_runs": runs,
                "offered_requests": len(int_burst),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
        "all_requests": _summarize_stream_group(
            all_interference_streams,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "measured_runs": runs,
                "offered_requests": len(all_interference_streams),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
    }
    comparison = summarize_interference_comparison(
        control_overall["victims"], interference_overall["victims"]
    )
    overall_statuses = [
        control_overall["victims"]["run_status"],
        interference_overall["victims"]["run_status"],
        control_overall["all_requests"]["run_status"],
        interference_overall["all_requests"]["run_status"],
    ]
    if any(status == "failed_population" for status in overall_statuses):
        overall_run_status = "failed_population"
    elif any(status == "failed_slo" for status in overall_statuses):
        overall_run_status = "failed_slo"
    elif any(status == "incomplete_slo_unavailable" for status in overall_statuses):
        overall_run_status = "incomplete_slo_unavailable"
    else:
        overall_run_status = "passed"

    finished = utc_now()
    return {
        "schema": INTERFERENCE_SCHEMA,
        "schema_extensions": [STREAM_DEADLINE_SCHEMA],
        "generated_utc": finished,
        "started_utc": started,
        "duration_seconds": clock() - started_monotonic,
        "note": note,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "target": {
            "label": endpoint.label,
            "base_url": endpoint.base_url,
            "model": model,
            "probe": probe,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": source_revision(revision),
            "selected_environment": selected_environment(),
        },
        "conditions": {
            "arrival_mode": "interference_burst",
            "control_mode": control_mode,
            "deadline_slo": norm_deadline_slo,
            "max_trace_events": max_trace_events,
            "cache_mode": cache_mode,
            "temperature": 0,
            "stream": True,
            "request_timeout_seconds": endpoint.timeout,
            "server_side_phase_attribution": dict(SERVER_PHASE_ATTRIBUTION_UNAVAILABLE),
        },
        "summary": {
            "control": control_overall,
            "interference": interference_overall,
            "comparison": comparison,
            "server_side_phase_attribution": dict(SERVER_PHASE_ATTRIBUTION_UNAVAILABLE),
            "run_status": overall_run_status,
        },
        "runs": measured,
    }


def _validate_interference_stream(
    stream: Any,
    expected_spec: dict[str, Any],
    source: str,
    *,
    role: str,
    deadline_slo: dict[str, float | None] | None,
) -> None:
    _require(isinstance(stream, dict), f"{source}: stream record is not an object")
    _require(
        stream.get("role") == role,
        f"{source}: stream role mismatch",
    )
    expected_id = (
        expected_spec["stream_id"] if role == "victim" else expected_spec["burst_id"]
    )
    _require(
        stream.get("stream_id") == expected_id,
        f"{source}: stream_id does not match workload",
    )
    _require(
        stream.get("max_tokens") == expected_spec["max_tokens"],
        f"{source}: stream max_tokens does not match workload",
    )
    _require(
        stream.get("prompt_sha256") == expected_spec["prompt_sha256"],
        f"{source}: stream prompt_sha256 does not match workload",
    )
    _require(
        stream.get("prompt_characters") == expected_spec["prompt_characters"],
        f"{source}: stream prompt_characters does not match workload",
    )
    _require(
        _summary_value_matches(
            expected_spec["planned_dispatch_offset_seconds"],
            stream.get("planned_dispatch_offset_seconds"),
        ),
        f"{source}: stream planned_dispatch_offset_seconds mismatch",
    )
    _require(
        _is_finite_number(stream.get("actual_dispatch_offset_seconds"))
        and stream["actual_dispatch_offset_seconds"] >= 0,
        f"{source}: stream actual_dispatch_offset_seconds is invalid",
    )
    _require(
        _is_finite_number(stream.get("completion_offset_seconds"))
        and stream["completion_offset_seconds"] >= stream["actual_dispatch_offset_seconds"],
        f"{source}: stream completion_offset_seconds is invalid",
    )
    outcome = stream.get("outcome")
    _require(
        outcome in REQUEST_OUTCOMES,
        f"{source}: stream outcome {outcome!r} is invalid",
    )
    _validate_stream_observation(stream, source, deadline_slo=deadline_slo)


def validate_interference_result(
    result: Any, source: str = "interference result"
) -> dict[str, Any]:
    _require(isinstance(result, dict), f"{source}: result is not an object")
    _require(
        result.get("schema") == INTERFERENCE_SCHEMA,
        f"{source}: schema is not {INTERFERENCE_SCHEMA}",
    )
    target = result.get("target")
    _require(
        isinstance(target, dict)
        and isinstance(target.get("label"), str)
        and bool(target["label"])
        and isinstance(target.get("base_url"), str)
        and bool(target["base_url"])
        and isinstance(target.get("model"), str)
        and bool(target["model"]),
        f"{source}: target metadata is invalid",
    )
    _require(
        isinstance(result.get("environment"), dict),
        f"{source}: environment is not an object",
    )
    fixture = validate_interference_workload(result.get("fixture"), source)
    fixture_sha256 = result.get("fixture_sha256")
    _require(
        isinstance(fixture_sha256, str)
        and len(fixture_sha256) == 64
        and canonical_sha256(fixture) == fixture_sha256,
        f"{source}: fixture digest does not match fixture",
    )
    workload = result.get("workload")
    _require(isinstance(workload, dict), f"{source}: workload is not an object")
    _require(
        isinstance(result.get("workload_sha256"), str)
        and canonical_sha256(workload) == result["workload_sha256"],
        f"{source}: workload digest does not match workload",
    )
    control_mode = workload.get("control_mode")
    _require(control_mode in CONTROL_MODES, f"{source}: control_mode is invalid")
    schedule_scale = workload.get("schedule_scale")
    _require(
        _is_finite_number(schedule_scale) and schedule_scale >= 0,
        f"{source}: schedule_scale is invalid",
    )
    measured_runs = workload.get("measured_runs")
    warmup_runs = workload.get("warmup_runs")
    settle_seconds = workload.get("settle_seconds")
    cache_mode = workload.get("cache_mode")
    _require(
        _is_int(measured_runs) and measured_runs > 0,
        f"{source}: measured_runs is invalid",
    )
    _require(
        _is_int(warmup_runs) and warmup_runs >= 0,
        f"{source}: warmup_runs is invalid",
    )
    _require(
        _is_finite_number(settle_seconds) and settle_seconds >= 0,
        f"{source}: settle_seconds is invalid",
    )
    _require(cache_mode in CACHE_MODES, f"{source}: cache_mode is invalid")
    expected_manifest = interference_workload_manifest(
        fixture,
        control_mode=control_mode,
        schedule_scale=schedule_scale,
        runs=measured_runs,
        warmup_runs=warmup_runs,
        settle_seconds=settle_seconds,
        cache_mode=cache_mode,
    )
    _require(
        _summary_value_matches(expected_manifest, workload),
        f"{source}: workload manifest does not match fixture",
    )

    conditions = result.get("conditions")
    _require(isinstance(conditions, dict), f"{source}: conditions are invalid")
    _require(
        conditions.get("arrival_mode") == "interference_burst",
        f"{source}: conditions.arrival_mode must be interference_burst",
    )
    _require(
        conditions.get("control_mode") == control_mode,
        f"{source}: conditions.control_mode mismatch",
    )
    _require(
        conditions.get("server_side_phase_attribution")
        == SERVER_PHASE_ATTRIBUTION_UNAVAILABLE,
        f"{source}: server_side_phase_attribution must be unavailable for client-only run",
    )
    deadline_slo = _validate_deadline_slo(conditions.get("deadline_slo"))
    _require(
        deadline_slo["ttft_slo_seconds"] is not None
        and deadline_slo["tbt_slo_seconds"] is not None
        and deadline_slo["max_gap_slo_seconds"] is not None,
        f"{source}: interference deadline_slo is incomplete",
    )

    runs = result.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == measured_runs,
        f"{source}: run count does not match workload",
    )
    all_control_streams: list[dict[str, Any]] = []
    all_interference_streams: list[dict[str, Any]] = []
    ctrl_spans: list[float] = []
    int_spans: list[float] = []
    for run_index, run in enumerate(runs):
        _require(isinstance(run, dict), f"{source}: run {run_index} is not an object")
        _require(run.get("index") == run_index, f"{source}: run index mismatch")
        for phase_name, burst_specs, burst_role in (
            ("control", workload["control_burst"], "control_burst"),
            ("interference", workload["interference_burst"], "interference_burst"),
        ):
            phase_obj = run.get(phase_name)
            _require(
                isinstance(phase_obj, dict),
                f"{source}: run {run_index} {phase_name} is not an object",
            )
            streams = phase_obj.get("streams")
            expected_count = len(workload["victim_streams"]) + len(burst_specs)
            _require(
                isinstance(streams, list) and len(streams) == expected_count,
                f"{source}: run {run_index} {phase_name} stream count mismatch",
            )
            for idx, vspec in enumerate(workload["victim_streams"]):
                _validate_interference_stream(
                    streams[idx],
                    vspec,
                    source,
                    role="victim",
                    deadline_slo=deadline_slo,
                )
            for bidx, bspec in enumerate(burst_specs):
                _validate_interference_stream(
                    streams[len(workload["victim_streams"]) + bidx],
                    bspec,
                    source,
                    role=burst_role,
                    deadline_slo=None,
                )
            recomputed_phase = summarize_interference_phase(
                phase_name,
                streams,
                control_mode=control_mode,
                warmup_runs=warmup_runs,
                deadline_slo=deadline_slo,
            )
            _require(
                _summary_value_matches(recomputed_phase, phase_obj.get("summary")),
                f"{source}: run {run_index} {phase_name} summary does not match streams",
            )
            if phase_name == "control":
                all_control_streams.extend(streams)
                if recomputed_phase["span_seconds"] is not None:
                    ctrl_spans.append(recomputed_phase["span_seconds"])
            else:
                all_interference_streams.extend(streams)
                if recomputed_phase["span_seconds"] is not None:
                    int_spans.append(recomputed_phase["span_seconds"])

        expected_cmp = summarize_interference_comparison(
            run["control"]["summary"]["victims"],
            run["interference"]["summary"]["victims"],
        )
        _require(
            _summary_value_matches(expected_cmp, run.get("comparison")),
            f"{source}: run {run_index} comparison does not match phase summaries",
        )

    summary = result.get("summary")
    _require(isinstance(summary, dict), f"{source}: summary is not an object")
    _require(
        summary.get("server_side_phase_attribution")
        == SERVER_PHASE_ATTRIBUTION_UNAVAILABLE,
        f"{source}: summary server_side_phase_attribution must be unavailable",
    )
    ctrl_victims = [s for s in all_control_streams if s["role"] == "victim"]
    int_victims = [s for s in all_interference_streams if s["role"] == "victim"]
    ctrl_burst = [s for s in all_control_streams if s["role"] != "victim"]
    int_burst = [s for s in all_interference_streams if s["role"] != "victim"]
    expected_control = {
        "victims": _summarize_stream_group(
            ctrl_victims,
            duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
            arrival_mode=f"control_{control_mode}",
            offered_load={
                "phase": "control",
                "control_mode": control_mode,
                "measured_runs": measured_runs,
                "victim_streams_per_run": len(fixture["victim_streams"]),
                "burst_requests_per_run": len(workload["control_burst"]),
                "offered_requests": len(ctrl_victims),
            },
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        ),
        "burst": (
            _summarize_stream_group(
                ctrl_burst,
                duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
                arrival_mode=f"control_{control_mode}",
                offered_load={
                    "phase": "control",
                    "role": "burst",
                    "measured_runs": measured_runs,
                    "offered_requests": len(ctrl_burst),
                },
                warmup_runs=warmup_runs,
                deadline_slo=None,
            )
            if ctrl_burst
            else None
        ),
        "all_requests": _summarize_stream_group(
            all_control_streams,
            duration_seconds=sum(ctrl_spans) if ctrl_spans else None,
            arrival_mode=f"control_{control_mode}",
            offered_load={
                "phase": "control",
                "control_mode": control_mode,
                "measured_runs": measured_runs,
                "offered_requests": len(all_control_streams),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
    }
    expected_interference = {
        "victims": _summarize_stream_group(
            int_victims,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "measured_runs": measured_runs,
                "victim_streams_per_run": len(fixture["victim_streams"]),
                "burst_requests_per_run": len(workload["interference_burst"]),
                "offered_requests": len(int_victims),
            },
            warmup_runs=warmup_runs,
            deadline_slo=deadline_slo,
        ),
        "burst": _summarize_stream_group(
            int_burst,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "role": "burst",
                "measured_runs": measured_runs,
                "offered_requests": len(int_burst),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
        "all_requests": _summarize_stream_group(
            all_interference_streams,
            duration_seconds=sum(int_spans) if int_spans else None,
            arrival_mode="interference_burst",
            offered_load={
                "phase": "interference",
                "measured_runs": measured_runs,
                "offered_requests": len(all_interference_streams),
            },
            warmup_runs=warmup_runs,
            deadline_slo=None,
        ),
    }
    _require(
        _summary_value_matches(expected_control, summary.get("control")),
        f"{source}: summary.control does not match its runs",
    )
    _require(
        _summary_value_matches(expected_interference, summary.get("interference")),
        f"{source}: summary.interference does not match its runs",
    )
    expected_comparison = summarize_interference_comparison(
        expected_control["victims"], expected_interference["victims"]
    )
    _require(
        _summary_value_matches(expected_comparison, summary.get("comparison")),
        f"{source}: summary.comparison does not match its runs",
    )
    statuses = [
        expected_control["victims"]["run_status"],
        expected_interference["victims"]["run_status"],
        expected_control["all_requests"]["run_status"],
        expected_interference["all_requests"]["run_status"],
    ]
    expected_status = (
        "failed_population"
        if any(s == "failed_population" for s in statuses)
        else (
            "failed_slo"
            if any(s == "failed_slo" for s in statuses)
            else (
                "incomplete_slo_unavailable"
                if any(s == "incomplete_slo_unavailable" for s in statuses)
                else "passed"
            )
        )
    )
    _require(
        summary.get("run_status") == expected_status,
        f"{source}: summary.run_status does not match its runs",
    )
    return result


def load_interference_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"{path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} is not JSON: {error}") from error
    return validate_interference_result(result, str(path))


def text_hashes(result: dict[str, Any]) -> dict[tuple[str, int], set[str]]:
    hashes: dict[tuple[str, int], set[str]] = {}
    for run in result.get("runs", []):
        if not isinstance(run, dict):
            continue
        mode = run.get("mode")
        for record in run.get("records", []):
            if not isinstance(record, dict):
                continue
            key = (str(mode), int(record["prompt_index"]))
            hashes.setdefault(key, set()).add(str(record["text_sha256"]))
    return hashes


def compare_results(
    results: list[dict[str, Any]], baseline_label: str | None = None
) -> dict[str, Any]:
    if len(results) < 2:
        raise BenchmarkError("compare needs at least two result files")
    results = [validate_result(result, "comparison input") for result in results]
    labels = [result["target"]["label"] for result in results]
    if len(set(labels)) != len(labels):
        raise BenchmarkError(f"duplicate target labels: {labels}")
    baseline_label = baseline_label or labels[0]
    if baseline_label not in labels:
        raise BenchmarkError(f"baseline label {baseline_label!r} is not in {labels}")
    ordered = [results[labels.index(baseline_label)]] + [
        result for result in results if result["target"]["label"] != baseline_label
    ]
    workload_hash = ordered[0]["workload_sha256"]
    for result in ordered[1:]:
        if result["workload_sha256"] != workload_hash:
            raise BenchmarkError(
                f"{result['target']['label']}: workload differs from {baseline_label}"
            )

    baseline = ordered[0]
    baseline_hashes = text_hashes(baseline)
    parity_mismatches: list[dict[str, Any]] = []
    deterministic = True
    for key, hashes in baseline_hashes.items():
        if len(hashes) != 1:
            deterministic = False
    for result in ordered[1:]:
        observed = text_hashes(result)
        for key in sorted(set(baseline_hashes) | set(observed)):
            reference = baseline_hashes.get(key, set())
            current = observed.get(key, set())
            if len(reference) != 1 or len(current) != 1:
                deterministic = False
            if reference != current:
                parity_mismatches.append({
                    "mode": key[0],
                    "prompt_index": key[1],
                    "target": result["target"]["label"],
                    "baseline_sha256": sorted(reference),
                    "observed_sha256": sorted(current),
                })

    metric_paths = [
        ("single", "delivered_tokens_per_second", "higher"),
        ("single", "decode_tokens_per_second", "higher"),
        ("single", "e2e_tokens_per_second", "higher"),
        ("single", "ttft_seconds_median", "lower"),
        ("single", "wall_seconds_p95", "lower"),
    ]
    for mode in baseline["summary"]:
        if mode.startswith("batch-"):
            metric_paths.extend([
                (mode, "delivered_tokens_per_second", "higher"),
                (mode, "aggregate_decode_tokens_per_second", "higher"),
                (mode, "per_user_decode_tokens_per_second", "higher"),
                (mode, "ttft_seconds_median", "lower"),
                (mode, "wall_seconds_p95", "lower"),
            ])

    ratios: dict[str, dict[str, Any]] = {}
    for result in ordered[1:]:
        label = result["target"]["label"]
        ratios[label] = {}
        for mode, metric, direction in metric_paths:
            base_value = baseline["summary"].get(mode, {}).get(metric)
            candidate_value = result["summary"].get(mode, {}).get(metric)
            ratio = None
            if (
                isinstance(base_value, (int, float))
                and isinstance(candidate_value, (int, float))
                and base_value != 0
            ):
                ratio = candidate_value / base_value
            ratios[label][f"{mode}.{metric}"] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "ratio": ratio,
                "direction": direction,
            }

    return {
        "schema": COMPARISON_SCHEMA,
        "generated_utc": utc_now(),
        "baseline": baseline_label,
        "targets": [result["target"]["label"] for result in ordered],
        "workload_sha256": workload_hash,
        "workload": baseline["workload"],
        "summaries": {
            result["target"]["label"]: result["summary"] for result in ordered
        },
        "ratios": ratios,
        "text_parity": {
            "deterministic": deterministic,
            "matched": not parity_mismatches and deterministic,
            "mismatches": parity_mismatches,
        },
        "provenance": {
            result["target"]["label"]: {
                "base_url": result["target"]["base_url"],
                "model": result["target"]["model"],
                "source_revision": result["environment"].get("source_revision"),
                "duration_seconds": result.get("duration_seconds"),
            }
            for result in ordered
        },
    }


def write_json(payload: dict[str, Any], path: str | None) -> None:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise BenchmarkError(f"refusing to write non-finite JSON: {error}") from error
    if path in (None, "-"):
        sys.stdout.write(text)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI endpoint for the device-free self-test."""

    server_version = "qwen38-benchmark-fake/1"

    def log_message(self, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "fake-model", "object": "model", "owned_by": "test"}
                    ],
                },
            )
            return
        if self.path in {"/healthz", "/health"}:
            self._json(200, {"status": "ok", "model": "fake-model"})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(request, dict):
            self._json(400, {"error": {"message": "body must be an object"}})
            return
        if self.path.endswith("/chat/completions"):
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": {"message": "messages are required"}})
                return
            prompt_text = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict)
            )
            prompt_tokens = max(1, len(prompt_text) // 4 + len(messages))
        elif self.path.endswith("/completions"):
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                self._json(400, {"error": {"message": "prompt is required"}})
                return
            prompt_tokens = max(1, len(prompt) // 4)
        else:
            self._json(404, {"error": {"message": "not found"}})
            return
        tokens = [" Paris", ".", " The", " capital"]
        if not request.get("stream"):
            self._json(
                200,
                {
                    "id": "cmpl-fake",
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "".join(tokens),
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(tokens),
                        "total_tokens": prompt_tokens + len(tokens),
                    },
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for index, token in enumerate(tokens):
            payload = {
                "id": "cmpl-fake",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": token,
                        "finish_reason": "length" if index == len(tokens) - 1 else None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.001)
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def self_test() -> None:
    corpus = {
        "id": "self-test",
        "prompts": ["The capital of France is", "Count from one to five in words:"],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        first = run_benchmark(
            Endpoint(
                label="fake-a",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        second = run_benchmark(
            Endpoint(
                label="fake-b",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        validate_result(first, "self-test")
        incomplete = json.loads(json.dumps(first))
        incomplete["runs"].pop()
        try:
            validate_result(incomplete, "self-test-incomplete")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an incomplete result passed validation")

        tampered = json.loads(json.dumps(first))
        tampered["summary"]["single"]["delivered_tokens_per_second"] = 1e9
        try:
            validate_result(tampered, "self-test-summary")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited summary passed validation")

        missing_target = json.loads(json.dumps(first))
        del missing_target["target"]["base_url"]
        try:
            validate_result(missing_target, "self-test-target")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a missing target URL passed validation")

        empty_text = json.loads(json.dumps(first))
        empty_text["runs"][0]["records"][0]["text_characters"] = 0
        try:
            validate_result(empty_text, "self-test-text")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an empty-text record passed validation")

        try:
            parse_headers(["X-Test: ☃"])
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a non-Latin-1 header was accepted")

        other_corpus = {
            "id": "self-test-other",
            "prompts": corpus["prompts"] + ["A different corpus changes the workload."],
        }
        different = run_benchmark(
            Endpoint(
                label="fake-other",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            other_corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        try:
            compare_results([first, different])
        except BenchmarkError as error:
            if "workload differs" not in str(error):
                raise BenchmarkError(
                    "self-test: a different corpus failed for the wrong reason"
                ) from error
        else:
            raise BenchmarkError("self-test: a different corpus was accepted")

        single = first["summary"]["single"]
        batch = first["summary"]["batch-2"]
        if not (single["ttft_seconds_median"] or 0) > 0:
            raise BenchmarkError("self-test: single TTFT was not measured")
        if not (single["decode_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: single decode rate was not measured")
        if not (batch["delivered_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: batch throughput was not measured")
        comparison = compare_results([first, second], baseline_label="fake-a")
        if not comparison["text_parity"]["matched"]:
            raise BenchmarkError("self-test: identical fake targets failed parity")
        if "fake-b" not in comparison["ratios"]:
            raise BenchmarkError("self-test: comparison omitted the candidate")

        plain = run_benchmark(
            Endpoint(
                label="fake-plain",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=False,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        if plain["summary"]["single"]["ttft_seconds_median"] is not None:
            raise BenchmarkError("self-test: non-stream TTFT was reported")
        if not (plain["summary"]["single"]["delivered_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: non-stream throughput was not measured")

        mismatch = json.loads(json.dumps(second))
        mismatch["runs"][0]["records"][0]["text_sha256"] = "0" * 64
        if compare_results([first, mismatch])["text_parity"]["matched"]:
            raise BenchmarkError("self-test: a text mismatch passed parity")
        wrong_workload = json.loads(json.dumps(second))
        wrong_workload["workload_sha256"] = "1" * 64
        try:
            compare_results([first, wrong_workload])
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a workload mismatch was accepted")

        trace = load_agentic_trace(DEFAULT_AGENTIC_TRACE)
        slo = {
            "ttft_p95_seconds": 5.0,
            "itl_p95_seconds": 5.0,
            "task_p95_seconds": 60.0,
        }
        agentic = run_agentic_benchmark(
            Endpoint(
                label="fake-agentic",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            trace,
            agents=[1, 2],
            runs=1,
            warmup_runs=0,
            settle_seconds=0.0,
            tool_wait_scale=0.0,
            slo=slo,
            cache_mode="disabled",
            power={"boundary": "cards", "watts": 10.0, "method": "self-test"},
            revision="self-test",
            note=None,
        )
        validate_agentic_result(agentic, "self-test-agentic")
        if agentic["summary"]["max_agents_meeting_all_slo"] != 2:
            raise BenchmarkError("self-test: agentic SLO maximum was not 2")
        if agentic["summary"]["saturated_agents"] != 2:
            raise BenchmarkError("self-test: agentic saturation width was not 2")
        if agentic["conditions"]["power"]["agents_at_slo_per_watt"] != 0.2:
            raise BenchmarkError("self-test: agents at SLO per watt was not derived")
        calls = [call for task in agentic["runs"][0]["tasks"] for call in task["calls"]]
        if not any((call["reused_prefix_tokens_estimate"] or 0) > 0 for call in calls):
            raise BenchmarkError("self-test: reused-prefix tokens were not recorded")
        if calls[0]["reused_prefix_characters"] != 0:
            raise BenchmarkError("self-test: the first call reported prefix reuse")
        if any(
            call["cache_confirmation"] != "client-prefix-overlap-only" for call in calls
        ):
            raise BenchmarkError("self-test: cache confirmation overclaimed")
        if agentic["workload"]["tool_wait_seconds_total"] != 0.0:
            raise BenchmarkError("self-test: scaled tool waits were not recorded")
        try:
            validate_result(agentic, "self-test-agentic-schema")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: agentic result passed the run schema")

        edited_call = json.loads(json.dumps(agentic))
        edited_call["runs"][0]["tasks"][0]["calls"][1][
            "reused_prefix_tokens_estimate"
        ] += 1
        try:
            validate_agentic_result(edited_call, "self-test-agentic-call")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited reused-token count passed")

        edited_curve = json.loads(json.dumps(agentic))
        edited_curve["summary"]["curve"][0]["ttft_seconds_p95"] = 0.0
        try:
            validate_agentic_result(edited_curve, "self-test-agentic-curve")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited SLO curve passed")

        edited_trace = json.loads(json.dumps(agentic))
        edited_trace["trace"]["system_prompt"] += " changed"
        try:
            validate_agentic_result(edited_trace, "self-test-agentic-trace")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited trace passed validation")

        missing_run = json.loads(json.dumps(agentic))
        missing_run["runs"].pop()
        try:
            validate_agentic_result(missing_run, "self-test-agentic-runs")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a missing agentic run passed")

        # Verify legacy v1 run JSON (without #17 extensions) still validates cleanly.
        legacy_v1 = json.loads(json.dumps(first))
        legacy_v1.pop("schema_extensions", None)
        legacy_v1.pop("conditions", None)
        for run in legacy_v1["runs"]:
            run["summary"].pop("span_seconds", None)
            for rec in run["records"]:
                for ext_key in ("stream_observation", "outcome", "outcome_reason", "error"):
                    rec.pop(ext_key, None)
        for mode_summary in legacy_v1["summary"].values():
            for ext_key in ("population_outcomes", "stream_deadlines", "run_status"):
                mode_summary.pop(ext_key, None)
        validate_result(legacy_v1, "self-test-legacy-v1")

        # Verify prefill-interference mode with both short_prompt and idle controls.
        interference_fixture = load_interference_workload(DEFAULT_INTERFERENCE_WORKLOAD)
        for ctrl_mode in ("short_prompt", "idle"):
            interference_res = run_prefill_interference_benchmark(
                Endpoint(
                    label=f"fake-interference-{ctrl_mode}",
                    base_url=base_url,
                    model=None,
                    api_key=None,
                    headers=(),
                    timeout=10.0,
                ),
                interference_fixture,
                control_mode=ctrl_mode,
                runs=1,
                warmup_runs=0,
                settle_seconds=0.0,
                schedule_scale=0.0,
                deadline_slo={
                    "ttft_slo_seconds": 5.0,
                    "tbt_slo_seconds": 5.0,
                    "max_gap_slo_seconds": 5.0,
                },
                cache_mode="disabled",
                revision="self-test",
                note=None,
            )
            validate_interference_result(
                interference_res, f"self-test-interference-{ctrl_mode}"
            )
            if (
                interference_res["summary"]["server_side_phase_attribution"]["status"]
                != "unavailable"
            ):
                raise BenchmarkError(
                    "self-test: interference phase attribution must be unavailable"
                )
            # _FakeHandler omits explicit token mapping, so token deadlines must be
            # unavailable and run_status must fail closed as incomplete_slo_unavailable.
            if (
                interference_res["summary"]["interference"]["victims"][
                    "stream_deadlines"
                ]["cumulative_token_deadline_compliance"]["status"]
                != "unavailable"
            ):
                raise BenchmarkError(
                    "self-test: unmapped interference streams must report unavailable token deadlines"
                )
            if interference_res["summary"]["run_status"] != "incomplete_slo_unavailable":
                raise BenchmarkError(
                    "self-test: unmapped interference run must fail closed as incomplete_slo_unavailable"
                )
            tampered_int = json.loads(json.dumps(interference_res))
            tampered_int["summary"]["comparison"]["ttft_seconds_p95"]["delta"] = 999.0
            try:
                validate_interference_result(
                    tampered_int, "self-test-interference-tampered"
                )
            except BenchmarkError:
                pass
            else:
                raise BenchmarkError(
                    "self-test: tampered interference comparison passed validation"
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    _self_test_pure_timestamp_reducer()
    _self_test_loopback_stream_scenarios()


class _ScriptedClock:
    """Deterministic monotonic clock for device-free loopback stream tests."""

    def __init__(
        self,
        values: list[float],
        *,
        on_call: Callable[[int, float], None] | None = None,
    ) -> None:
        self._values = list(values)
        self._index = 0
        self._on_call = on_call

    def __call__(self) -> float:
        idx = self._index
        value = self._values[min(idx, len(self._values) - 1)]
        self._index += 1
        if self._on_call is not None:
            self._on_call(idx, value)
        return value


def _self_test_pure_timestamp_reducer() -> None:
    """Deterministic #17 witness and cumulative-slack separation tests."""
    smooth = [0.10, 0.35, 0.60, 0.85, 1.10]
    burst = [0.10, 1.07, 1.08, 1.09, 1.10]

    smooth_res = evaluate_stream_timestamps(
        smooth,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
        token_timestamps_seconds=smooth,
        token_mapping_provenance="explicit_timestamps",
    )
    burst_res = evaluate_stream_timestamps(
        burst,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
        token_timestamps_seconds=burst,
        token_mapping_provenance="explicit_timestamps",
    )

    # Identical TTFT (0.10) and identical mean ITL (0.25).
    if not math.isclose(smooth_res["ttft_seconds"], 0.10, abs_tol=1e-12):
        raise BenchmarkError("self-test: smooth TTFT mismatch")
    if not math.isclose(burst_res["ttft_seconds"], 0.10, abs_tol=1e-12):
        raise BenchmarkError("self-test: burst TTFT mismatch")
    if not math.isclose(smooth_res["mean_itl_seconds"], 0.25, abs_tol=1e-12):
        raise BenchmarkError("self-test: smooth mean ITL mismatch")
    if not math.isclose(burst_res["mean_itl_seconds"], 0.25, abs_tol=1e-12):
        raise BenchmarkError("self-test: burst mean ITL mismatch")

    # Max gap 0.25 vs 0.97.
    if not math.isclose(
        smooth_res["content_event_gaps"]["event_gap_seconds_max"], 0.25, abs_tol=1e-12
    ):
        raise BenchmarkError("self-test: smooth max gap mismatch")
    if not math.isclose(
        burst_res["content_event_gaps"]["event_gap_seconds_max"], 0.97, abs_tol=1e-12
    ):
        raise BenchmarkError("self-test: burst max gap mismatch")

    # Cumulative deadlines [0.20, 0.50, 0.80, 1.10, 1.40]: smooth meets all 5;
    # burst misses tokens 2 and 3.
    if smooth_res["max_gap_compliance"]["max_gap_within_budget"] is not True:
        raise BenchmarkError("self-test: smooth should pass max_gap_within_budget")
    if smooth_res["token_deadline_compliance"]["every_token_meets_deadline"] is not True:
        raise BenchmarkError("self-test: smooth should meet all 5 token deadlines")
    if smooth_res["token_deadline_compliance"]["met_tokens"] != 5:
        raise BenchmarkError("self-test: smooth met_tokens must be 5")

    if burst_res["max_gap_compliance"]["max_gap_within_budget"] is not False:
        raise BenchmarkError("self-test: burst should fail max_gap_within_budget")
    if burst_res["token_deadline_compliance"]["every_token_meets_deadline"] is not False:
        raise BenchmarkError("self-test: burst must fail cumulative token deadlines")
    if burst_res["token_deadline_compliance"]["missed_token_ordinals"] != [2, 3]:
        raise BenchmarkError(
            "self-test: burst must miss exactly tokens [2, 3], got "
            f"{burst_res['token_deadline_compliance']['missed_token_ordinals']}"
        )

    # Cumulative slack separation: early delivery creates slack so cumulative
    # deadlines pass even though a single gap (0.35) exceeds TBT_SLO (0.30).
    slack_stream = [0.05, 0.10, 0.45, 0.70]
    slack_res = evaluate_stream_timestamps(
        slack_stream,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
        token_timestamps_seconds=slack_stream,
        token_mapping_provenance="explicit_timestamps",
    )
    if slack_res["max_gap_compliance"]["max_gap_within_budget"] is not False:
        raise BenchmarkError("self-test: slack stream must fail max_gap_within_budget")
    if slack_res["token_deadline_compliance"]["every_token_meets_deadline"] is not True:
        raise BenchmarkError(
            "self-test: slack stream must pass cumulative token deadlines"
        )

    # Reverse separation: max_gap within budget (0.27 <= 0.30) while late TTFT
    # causes tokens 1 and 2 to miss cumulative deadlines [0.20, 0.50, 0.80, 1.10].
    late_first = [0.25, 0.52, 0.79, 1.06]
    late_res = evaluate_stream_timestamps(
        late_first,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
        token_timestamps_seconds=late_first,
        token_mapping_provenance="explicit_timestamps",
    )
    if late_res["max_gap_compliance"]["max_gap_within_budget"] is not True:
        raise BenchmarkError("self-test: late_first must pass max_gap_within_budget")
    if late_res["token_deadline_compliance"]["every_token_meets_deadline"] is not False:
        raise BenchmarkError("self-test: late_first must fail cumulative deadlines")
    if late_res["token_deadline_compliance"]["missed_token_ordinals"] != [1, 2]:
        raise BenchmarkError("self-test: late_first must miss tokens [1, 2]")

    # Unmapped stream must report unavailable, never passed or 0.
    unmapped = evaluate_stream_timestamps(
        smooth,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
    )
    if unmapped["token_deadline_compliance"]["status"] != "unavailable":
        raise BenchmarkError("self-test: unmapped token deadlines must be unavailable")
    if unmapped["token_deadline_compliance"]["every_token_meets_deadline"] is not None:
        raise BenchmarkError(
            "self-test: unmapped every_token_meets_deadline must be None"
        )

    # Truncated and non-monotonic traces must fail closed.
    truncated = evaluate_stream_timestamps(
        smooth,
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
        token_timestamps_seconds=smooth,
        token_mapping_provenance="explicit_timestamps",
        trace_truncated=True,
    )
    if (
        truncated["content_event_gaps"]["status"] != "invalid_truncated"
        or truncated["max_gap_compliance"]["max_gap_within_budget"] is not False
        or truncated["token_deadline_compliance"]["every_token_meets_deadline"]
        is not False
    ):
        raise BenchmarkError("self-test: trace overflow did not invalidate compliance")

    non_mono = evaluate_stream_timestamps(
        [0.10, 0.40, 0.30],
        ttft_slo_seconds=0.20,
        tbt_slo_seconds=0.30,
        max_gap_slo_seconds=0.30,
    )
    if (
        non_mono["timing_valid"] is not False
        or non_mono["content_event_gaps"]["status"] != "invalid_timing"
        or non_mono["max_gap_compliance"]["max_gap_within_budget"] is not False
    ):
        raise BenchmarkError("self-test: non-monotonic timestamps were not rejected")


class _ScenarioHandler(BaseHTTPRequestHandler):
    """Configurable SSE loopback server for deterministic #17 scenario tests."""

    server_version = "qwen38-benchmark-scenario/1"

    def log_message(self, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse_raw(self, text: str) -> None:
        try:
            self.wfile.write(text.encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse_data(self, payload: dict[str, Any]) -> None:
        self._sse_raw(f"data: {json.dumps(payload)}\n\n")

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(
                200,
                {"object": "list", "data": [{"id": "scenario-model", "object": "model"}]},
            )
            return
        if self.path in {"/healthz", "/health"}:
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        _request = json.loads(self.rfile.read(length) or b"{}")
        scenario = self.headers.get("X-Scenario", "token_ids_5")

        if scenario == "http_503":
            self._json(503, {"error": {"message": "overloaded"}})
            return

        if scenario == "stall_timeout":
            self._start_sse()
            release_event = getattr(self.server, "timeout_release_event", None)
            if release_event is not None:
                release_event.wait(timeout=5.0)
            return

        if scenario == "flaky_agentic":
            counter = getattr(self.server, "flaky_counter", 0)
            self.server.flaky_counter = counter + 1
            if counter == 1:
                self._json(503, {"error": {"message": "injected failure"}})
                return
            scenario = "token_ids_5"

        self._start_sse()

        if scenario == "token_ids_5":
            tokens = ["one", " two", " three", " four", " five"]
            for idx, text in enumerate(tokens):
                self._sse_data({
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text, "token_ids": [100 + idx]},
                        "finish_reason": "stop" if idx == len(tokens) - 1 else None,
                    }]
                })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "cumulative_tokens_5":
            tokens = ["a", "b", "c", "d", "e"]
            for idx, text in enumerate(tokens):
                self._sse_data({
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "cumulative_completion_tokens": idx + 1,
                        "finish_reason": "stop" if idx == len(tokens) - 1 else None,
                    }]
                })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "token_ids_4":
            tokens = ["a", "b", "c", "d"]
            for idx, text in enumerate(tokens):
                self._sse_data({
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text, "token_ids": [200 + idx]},
                        "finish_reason": "stop" if idx == len(tokens) - 1 else None,
                    }]
                })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "coalesced_unmapped":
            self._sse_data({"choices": [{"index": 0, "delta": {"content": "one two"}, "finish_reason": None}]})
            self._sse_data({"choices": [{"index": 0, "delta": {"content": " three four five"}, "finish_reason": "stop"}]})
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "five_chunks_unmapped":
            for idx, text in enumerate(["a", "b", "c", "d", "e"]):
                self._sse_data({
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": "stop" if idx == 4 else None,
                    }]
                })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "coalesced_token_ids":
            self._sse_data({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "one two", "token_ids": [10, 11]},
                    "finish_reason": None,
                }]
            })
            self._sse_data({
                "choices": [{
                    "index": 0,
                    "delta": {"content": " three four five", "token_ids": [12, 13, 14]},
                    "finish_reason": "stop",
                }]
            })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "heartbeat_role_empty_usage":
            self._sse_raw(": keep-alive-1\n\n")
            self._sse_data({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            self._sse_data({"choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
            self._sse_data({"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]})
            self._sse_raw(": keep-alive-2\n\n")
            self._sse_data({"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]})
            self._sse_data({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "one_token_mapped":
            self._sse_data({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "ok", "token_ids": [42]},
                    "finish_reason": "stop",
                }]
            })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "one_token_unmapped":
            self._sse_data({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "ok"},
                    "finish_reason": "stop",
                }]
            })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
            self._sse_raw("data: [DONE]\n\n")
            return

        if scenario == "eof_without_completion":
            self._sse_data({"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]})
            self._sse_data({"choices": [{"index": 0, "delta": {"content": " cut"}, "finish_reason": None}]})
            return

        if scenario == "token_ids_count_mismatch":
            self._sse_data({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "one", "token_ids": [1]},
                    "finish_reason": "stop",
                }]
            })
            self._sse_data({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}})
            self._sse_raw("data: [DONE]\n\n")
            return

        self._sse_raw("data: [DONE]\n\n")


def _self_test_loopback_stream_scenarios() -> None:
    """Deterministic injected-clock loopback SSE tests for all #17 requirements."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScenarioHandler)
    server.daemon_threads = True
    timeout_release = threading.Event()
    server.timeout_release_event = timeout_release
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    slo = {
        "ttft_slo_seconds": 0.20,
        "tbt_slo_seconds": 0.30,
        "max_gap_slo_seconds": 0.30,
    }
    base_url = f"http://127.0.0.1:{server.server_port}"

    def run_scenario(
        scenario: str,
        clock: Callable[[], float],
        *,
        max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
        cancel_event: threading.Event | None = None,
        deadline_slo: dict[str, float | None] | None = slo,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        endpoint = Endpoint(
            label=f"scen-{scenario}",
            base_url=base_url,
            model="scenario-model",
            api_key=None,
            headers=(("X-Scenario", scenario),),
            timeout=timeout,
        )
        conn = open_connection(endpoint)
        try:
            start = clock()
            record, _usage = observe_stream_completion(
                conn,
                endpoint,
                endpoint.api_path("/chat/completions"),
                {
                    "model": "scenario-model",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 16,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                scenario,
                start,
                clock=clock,
                max_trace_events=max_trace_events,
                deadline_slo=deadline_slo,
                cancel_event=cancel_event,
            )
            return record
        finally:
            conn.close()

    try:
        # 1. Smooth delivery with explicit endpoint_token_ids.
        smooth_rec = run_scenario(
            "token_ids_5",
            _ScriptedClock([0.0, 0.10, 0.35, 0.60, 0.85, 1.10, 1.15]),
        )
        if smooth_rec["outcome"] != "completed":
            raise BenchmarkError("self-test: smooth loopback stream did not complete")
        smooth_obs = smooth_rec["stream_observation"]
        if smooth_obs["token_mapping"]["provenance"] != "endpoint_token_ids":
            raise BenchmarkError("self-test: expected endpoint_token_ids provenance")
        if smooth_obs["max_gap_compliance"]["max_gap_within_budget"] is not True:
            raise BenchmarkError("self-test: smooth loopback failed max_gap_within_budget")
        if smooth_obs["token_deadline_compliance"]["every_token_meets_deadline"] is not True:
            raise BenchmarkError("self-test: smooth loopback failed token deadlines")
        if "text" in smooth_rec:
            raise BenchmarkError("self-test: raw response text must not be stored")

        # 2. Stalled/burst delivery with explicit endpoint_cumulative_tokens.
        burst_rec = run_scenario(
            "cumulative_tokens_5",
            _ScriptedClock([0.0, 0.10, 1.07, 1.08, 1.09, 1.10, 1.15]),
        )
        burst_obs = burst_rec["stream_observation"]
        if burst_rec["outcome"] != "completed":
            raise BenchmarkError("self-test: burst loopback stream did not complete")
        if not math.isclose(burst_rec["ttft_seconds"], 0.10, abs_tol=1e-12):
            raise BenchmarkError("self-test: burst loopback TTFT mismatch")
        if burst_obs["token_mapping"]["provenance"] != "endpoint_cumulative_tokens":
            raise BenchmarkError("self-test: expected endpoint_cumulative_tokens")
        if burst_obs["max_gap_compliance"]["max_gap_within_budget"] is not False:
            raise BenchmarkError("self-test: burst loopback must fail max_gap_within_budget")
        if burst_obs["token_deadline_compliance"]["every_token_meets_deadline"] is not False:
            raise BenchmarkError("self-test: burst loopback must fail token deadlines")
        if burst_obs["token_deadline_compliance"]["missed_token_ordinals"] != [2, 3]:
            raise BenchmarkError("self-test: burst loopback missed ordinals must be [2, 3]")

        # 3. Cumulative slack passes even when max_gap > TBT_SLO.
        slack_rec = run_scenario(
            "token_ids_4",
            _ScriptedClock([0.0, 0.05, 0.10, 0.45, 0.70, 0.75]),
        )
        slack_obs = slack_rec["stream_observation"]
        if slack_obs["max_gap_compliance"]["max_gap_within_budget"] is not False:
            raise BenchmarkError("self-test: slack loopback must fail max_gap_within_budget")
        if slack_obs["token_deadline_compliance"]["every_token_meets_deadline"] is not True:
            raise BenchmarkError("self-test: slack loopback must pass token deadlines")

        # 4. Coalesced / unmapped deltas: token deadlines MUST be unavailable.
        coalesced_unmapped = run_scenario(
            "coalesced_unmapped",
            _ScriptedClock([0.0, 0.10, 0.30, 0.35]),
        )
        cu_obs = coalesced_unmapped["stream_observation"]
        if cu_obs["content_event_gaps"]["gap_count"] != 1:
            raise BenchmarkError("self-test: coalesced stream must report 1 event gap")
        if cu_obs["token_deadline_compliance"]["status"] != "unavailable":
            raise BenchmarkError("self-test: coalesced unmapped must report unavailable")
        if cu_obs["token_deadline_compliance"]["every_token_meets_deadline"] is not None:
            raise BenchmarkError("self-test: unavailable compliance must be None")

        # Even when stream_chunks == completion_tokens, unmapped deltas stay unavailable.
        five_unmapped = run_scenario(
            "five_chunks_unmapped",
            _ScriptedClock([0.0, 0.10, 0.35, 0.60, 0.85, 1.10, 1.15]),
        )
        if (
            five_unmapped["stream_observation"]["token_deadline_compliance"]["status"]
            != "unavailable"
        ):
            raise BenchmarkError(
                "self-test: unmapped 1-chunk-per-token stream must not equate chunks with tokens"
            )

        # 5. Coalesced deltas WITH explicit endpoint_token_ids (no backward interpolation!).
        coalesced_mapped = run_scenario(
            "coalesced_token_ids",
            _ScriptedClock([0.0, 0.10, 0.85, 0.90]),
        )
        cm_tdc = coalesced_mapped["stream_observation"]["token_deadline_compliance"]
        # Tokens 1,2 at 0.10 (deadlines 0.20, 0.50 -> pass);
        # Tokens 3,4,5 at 0.85 (deadlines 0.80, 1.10, 1.40 -> token 3 misses!).
        if cm_tdc["status"] != "failed" or cm_tdc["missed_token_ordinals"] != [3]:
            raise BenchmarkError(
                "self-test: coalesced mapped tokens must evaluate at event arrival time without interpolation"
            )

        # 6. Empty / heartbeat / role-only / usage-only / [DONE] separation.
        noisy = run_scenario(
            "heartbeat_role_empty_usage",
            _ScriptedClock([0.0, 0.12, 0.28, 0.30]),
        )
        noisy_obs = noisy["stream_observation"]
        expected_counts = {
            "content": 2,
            "heartbeat": 2,
            "role_only": 1,
            "usage_only": 1,
            "empty": 2,
            "done": 1,
        }
        if noisy_obs["event_counts"] != expected_counts:
            raise BenchmarkError(
                f"self-test: event_counts mismatch: {noisy_obs['event_counts']} != {expected_counts}"
            )
        if not math.isclose(noisy["ttft_seconds"], 0.12, abs_tol=1e-12):
            raise BenchmarkError("self-test: heartbeat/role events polluted TTFT")
        if len(noisy_obs["content_event_offsets_seconds"]) != 2:
            raise BenchmarkError("self-test: non-content events entered content offsets")

        # 7. One-token outputs (mapped pass, mapped miss, and unmapped).
        one_pass = run_scenario("one_token_mapped", _ScriptedClock([0.0, 0.12, 0.15]))
        one_pass_obs = one_pass["stream_observation"]
        if (
            one_pass["outcome"] != "completed"
            or one_pass_obs["content_event_gaps"]["status"] != "single_content_event"
            or one_pass_obs["max_gap_compliance"]["max_gap_within_budget"] is not True
            or one_pass_obs["token_deadline_compliance"]["every_token_meets_deadline"]
            is not True
        ):
            raise BenchmarkError("self-test: one-token mapped pass failed")

        one_miss = run_scenario("one_token_mapped", _ScriptedClock([0.0, 0.25, 0.30]))
        if (
            one_miss["stream_observation"]["token_deadline_compliance"][
                "every_token_meets_deadline"
            ]
            is not False
            or one_miss["stream_observation"]["token_deadline_compliance"][
                "missed_token_ordinals"
            ]
            != [1]
        ):
            raise BenchmarkError("self-test: one-token mapped miss did not fail token 1")

        one_unmapped = run_scenario(
            "one_token_unmapped", _ScriptedClock([0.0, 0.12, 0.15])
        )
        if (
            one_unmapped["stream_observation"]["token_deadline_compliance"]["status"]
            != "unavailable"
        ):
            raise BenchmarkError("self-test: one-token unmapped must be unavailable")

        # 8. EOF without valid completion.
        eof_rec = run_scenario(
            "eof_without_completion", _ScriptedClock([0.0, 0.10, 0.20, 0.25])
        )
        if (
            eof_rec["outcome"] != "incomplete"
            or eof_rec["outcome_reason"] != "eof_without_valid_completion"
        ):
            raise BenchmarkError("self-test: premature EOF was not marked incomplete")

        # 9. Cancellation mid-stream.
        cancel_ev = threading.Event()

        def cancel_after_first(call_idx: int, _val: float) -> None:
            if call_idx == 1:
                cancel_ev.set()

        cancel_rec = run_scenario(
            "token_ids_5",
            _ScriptedClock([0.0, 0.10, 0.20, 0.30], on_call=cancel_after_first),
            cancel_event=cancel_ev,
        )
        if (
            cancel_rec["outcome"] != "incomplete"
            or cancel_rec["outcome_reason"] != "cancelled"
        ):
            raise BenchmarkError("self-test: cancelled stream was not classified properly")

        # 10. Trace overflow (max_trace_events=3 on 5-event stream).
        overflow_rec = run_scenario(
            "token_ids_5",
            _ScriptedClock([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]),
            max_trace_events=3,
        )
        ov_obs = overflow_rec["stream_observation"]
        if (
            overflow_rec["outcome"] != "unobservable"
            or overflow_rec["outcome_reason"] != "trace_overflow"
            or ov_obs["trace_truncated"] is not True
            or ov_obs["overflow"] is not True
            or ov_obs["recorded_trace_events"] != 3
            or ov_obs["dropped_trace_events"] != 2
            or ov_obs["content_event_gaps"]["status"] != "invalid_truncated"
            or ov_obs["max_gap_compliance"]["max_gap_within_budget"] is not False
            or ov_obs["token_deadline_compliance"]["every_token_meets_deadline"]
            is not False
        ):
            raise BenchmarkError("self-test: trace overflow handling failed")

        # 11. Invalid / non-monotonic / non-finite timing metadata & mapping mismatch.
        non_mono_rec = run_scenario(
            "token_ids_5",
            _ScriptedClock([0.0, 0.10, 0.40, 0.30, 0.50, 0.60, 0.65]),
        )
        if (
            non_mono_rec["outcome"] != "unobservable"
            or non_mono_rec["stream_observation"]["timing_valid"] is not False
            or non_mono_rec["stream_observation"]["content_event_gaps"]["status"]
            != "invalid_timing"
        ):
            raise BenchmarkError("self-test: non-monotonic loopback clock not invalidated")

        nan_rec = run_scenario(
            "token_ids_5",
            _ScriptedClock([0.0, 0.10, float("nan"), 0.30, 0.40, 0.50, 0.55]),
        )
        if (
            nan_rec["outcome"] != "unobservable"
            or nan_rec["stream_observation"]["timing_valid"] is not False
        ):
            raise BenchmarkError("self-test: NaN loopback clock not invalidated")
        json.dumps(nan_rec, allow_nan=False)

        mismatch_map = run_scenario(
            "token_ids_count_mismatch", _ScriptedClock([0.0, 0.10, 0.15])
        )
        if (
            mismatch_map["stream_observation"]["token_deadline_compliance"]["status"]
            != "unavailable"
            or mismatch_map["stream_observation"]["token_mapping"]["reason"]
            != "invalid_token_mapping_count_mismatch"
        ):
            raise BenchmarkError("self-test: token_ids count mismatch was not rejected")

        # 12. HTTP 503 rejection, read timeout, and mixed failure-retaining rollup.
        rejected_rec = run_scenario("http_503", _ScriptedClock([0.0, 0.05]))
        if rejected_rec["outcome"] != "rejected":
            raise BenchmarkError("self-test: HTTP 503 was not classified as rejected")

        timeout_rec = run_scenario(
            "stall_timeout", _ScriptedClock([0.0, 0.05]), timeout=0.05
        )
        if (
            timeout_rec["outcome"] != "timed_out"
            or timeout_rec["outcome_reason"] != "request_timeout"
        ):
            raise BenchmarkError("self-test: stalled socket was not classified as timed_out")

        mixed_population = [
            smooth_rec,
            burst_rec,
            rejected_rec,
            timeout_rec,
            eof_rec,
            cancel_rec,
            overflow_rec,
        ]
        mixed_summary = summarize_population_and_deadlines(
            mixed_population,
            duration_seconds=2.0,
            arrival_mode="closed_loop",
            offered_load={"offered_requests": len(mixed_population)},
            warmup_runs=0,
            deadline_slo=slo,
        )
        pop = mixed_summary["population_outcomes"]
        if (
            pop["offered"] != 7
            or pop["admitted"] != "unknown"
            or pop["completed"] != 2
            or pop["rejected"] != 1
            or pop["timed_out"] != 1
            or pop["incomplete"] != 2
            or pop["unobservable"] != 1
            or pop["cancelled"] != 1
            or pop["run_status"] != "failed_population"
        ):
            raise BenchmarkError(f"self-test: mixed population counts wrong: {pop}")
        # Verify explicit event_weighted vs request_weighted sample counts.
        ev_w = mixed_summary["stream_deadlines"]["event_weighted"]
        req_w = mixed_summary["stream_deadlines"]["request_weighted"]
        if (
            ev_w["weighting"] != "event_weighted"
            or ev_w["sample_count"] != 8
            or ev_w["request_count"] != 2
            or ev_w["excluded_or_unobservable_requests"] != 5
            or req_w["weighting"] != "request_weighted"
            or req_w["sample_count"] != 2
            or req_w["total_offered_requests"] != 7
        ):
            raise BenchmarkError(
                f"self-test: weighting sample counts mismatch: {ev_w} / {req_w}"
            )
        mg = mixed_summary["stream_deadlines"]["max_gap_slo_compliance"]
        if mg["denominator_requests"] != 7 or not math.isclose(
            mg["compliance_rate"], 1 / 7, abs_tol=1e-12
        ):
            raise BenchmarkError(
                "self-test: max-gap compliance denominator dropped failed/unobservable requests"
            )
        td = mixed_summary["stream_deadlines"]["cumulative_token_deadline_compliance"]
        if (
            td["denominator_requests"] != 7
            or td["every_token_meets_deadline"] is not False
            or not math.isclose(
                td["request_weighted_compliance_rate"], 1 / 7, abs_tol=1e-12
            )
        ):
            raise BenchmarkError(
                "self-test: cumulative deadline denominator dropped failed/unobservable requests"
            )

        # Verify agentic mode retains failures and fails closed on population errors.
        trace = load_agentic_trace(DEFAULT_AGENTIC_TRACE)
        server.flaky_counter = 0
        flaky_agentic = run_agentic_benchmark(
            Endpoint(
                label="flaky-agentic",
                base_url=base_url,
                model="scenario-model",
                api_key=None,
                headers=(("X-Scenario", "flaky_agentic"),),
                timeout=10.0,
            ),
            trace,
            agents=[1],
            runs=1,
            warmup_runs=0,
            settle_seconds=0.0,
            tool_wait_scale=0.0,
            slo={
                "ttft_p95_seconds": 5.0,
                "itl_p95_seconds": 5.0,
                "task_p95_seconds": 60.0,
            },
            cache_mode="disabled",
            power={"boundary": "unspecified", "watts": None, "method": None},
            revision="self-test",
            note=None,
            deadline_slo={
                "ttft_slo_seconds": 5.0,
                "tbt_slo_seconds": 5.0,
                "max_gap_slo_seconds": 5.0,
            },
        )
        validate_agentic_result(flaky_agentic, "self-test-flaky-agentic")
        if flaky_agentic["summary"]["population_outcomes"]["rejected"] != 1:
            raise BenchmarkError("self-test: flaky agentic run dropped rejected call")
        if flaky_agentic["summary"]["run_status"] != "failed_population":
            raise BenchmarkError("self-test: flaky agentic run did not fail closed")
        if flaky_agentic["summary"]["max_agents_meeting_all_slo"] is not None:
            raise BenchmarkError(
                "self-test: flaky agentic run allowed failed width to meet SLO"
            )

        # Verify interference mode passes when explicit token mapping is provided.
        mapped_fixture = load_interference_workload(DEFAULT_INTERFERENCE_WORKLOAD)
        mapped_interference = run_prefill_interference_benchmark(
            Endpoint(
                label="mapped-interference",
                base_url=base_url,
                model="scenario-model",
                api_key=None,
                headers=(("X-Scenario", "token_ids_5"),),
                timeout=10.0,
            ),
            mapped_fixture,
            control_mode="short_prompt",
            runs=1,
            warmup_runs=0,
            settle_seconds=0.0,
            schedule_scale=0.0,
            deadline_slo={
                "ttft_slo_seconds": 5.0,
                "tbt_slo_seconds": 5.0,
                "max_gap_slo_seconds": 5.0,
            },
            cache_mode="disabled",
            revision="self-test",
            note=None,
        )
        validate_interference_result(mapped_interference, "self-test-mapped-interference")
        if mapped_interference["summary"]["run_status"] != "passed":
            raise BenchmarkError(
                "self-test: mapped interference run within SLO did not pass"
            )
    finally:
        timeout_release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def parse_headers(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    headers = []
    for value in values or []:
        name, separator, content = value.partition(":")
        if not separator or not name.strip():
            raise BenchmarkError(f"invalid --header {value!r}; expected NAME: VALUE")
        name = name.strip()
        content = content.strip()
        try:
            validate_header(name, content)
        except BenchmarkError as error:
            raise BenchmarkError(f"invalid --header {value!r}: {error}") from error
        headers.append((name, content))
    return tuple(headers)


def parse_concurrency(value: str) -> list[int]:
    widths = []
    for item in value.split(","):
        try:
            width = int(item)
        except ValueError as error:
            raise BenchmarkError(f"invalid concurrency {item!r}") from error
        if width <= 1:
            raise BenchmarkError("concurrency widths must be greater than one")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise BenchmarkError("at least one concurrency width is required")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeatable OpenAI-compatible benchmark for Qwen serving stacks"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the device-free fake-server self-test and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="measure one endpoint")
    run.add_argument("--label", required=True, help="short target name, e.g. ours")
    run.add_argument("--base-url", required=True, help="http://host:port")
    run.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served model name; discovered from /v1/models when omitted",
    )
    run.add_argument(
        "--api-key",
        default=os.environ.get("QWEN38_BENCH_API_KEY"),
        help="bearer token, never written to the result (default: QWEN38_BENCH_API_KEY)",
    )
    run.add_argument(
        "--header",
        action="append",
        help="additional request header as NAME: VALUE (repeatable)",
    )
    run.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help="corpus JSON (default: the packaged B=8 corpus)",
    )
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument(
        "--concurrency",
        default="8",
        help="comma-separated batch widths greater than one (default: 8)",
    )
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--warmup-runs", type=int, default=1)
    run.add_argument(
        "--no-stream",
        action="store_true",
        help="do not stream; TTFT and decode rate are then unavailable",
    )
    run.add_argument("--settle-seconds", type=float, default=1.0)
    run.add_argument("--request-timeout", type=float, default=600.0)
    run.add_argument(
        "--max-trace-events",
        type=int,
        default=DEFAULT_MAX_TRACE_EVENTS,
        help=f"maximum per-stream SSE content events recorded (default: {DEFAULT_MAX_TRACE_EVENTS})",
    )
    run.add_argument(
        "--ttft-slo",
        type=float,
        help="optional cumulative-deadline TTFT SLO in seconds (requires --tbt-slo)",
    )
    run.add_argument(
        "--tbt-slo",
        type=float,
        help="optional cumulative-deadline TBT SLO in seconds (requires --ttft-slo)",
    )
    run.add_argument(
        "--max-gap-slo",
        type=float,
        help="optional per-stream maximum content-event gap SLO in seconds",
    )
    run.add_argument(
        "--source-revision",
        help="repository revision recorded in the result; auto-detected when omitted",
    )
    run.add_argument("--note", help="free-form context recorded in the result")
    run.add_argument(
        "--output",
        default="-",
        help="result path, or - for stdout (default: -)",
    )

    agentic = subparsers.add_parser(
        "agentic",
        help="measure concurrent closed-loop agents on a frozen multi-call trace",
    )
    agentic.add_argument("--label", required=True, help="short target name, e.g. ours")
    agentic.add_argument("--base-url", required=True, help="http://host:port")
    agentic.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served model name; discovered from /v1/models when omitted",
    )
    agentic.add_argument(
        "--api-key",
        default=os.environ.get("QWEN38_BENCH_API_KEY"),
        help="bearer token, never written to the result (default: QWEN38_BENCH_API_KEY)",
    )
    agentic.add_argument(
        "--header",
        action="append",
        help="additional request header as NAME: VALUE (repeatable)",
    )
    agentic.add_argument(
        "--trace",
        default=str(DEFAULT_AGENTIC_TRACE),
        help="agentic trace JSON (default: the packaged eight-call trace)",
    )
    agentic.add_argument(
        "--agents",
        default="1,2,4,8",
        help="comma-separated concurrent agent counts (default: 1,2,4,8)",
    )
    agentic.add_argument("--runs", type=int, default=2)
    agentic.add_argument("--warmup-runs", type=int, default=1)
    agentic.add_argument("--settle-seconds", type=float, default=1.0)
    agentic.add_argument(
        "--tool-wait-scale",
        type=float,
        default=1.0,
        help="multiply every trace tool wait; recorded in the workload digest",
    )
    agentic.add_argument(
        "--ttft-p95",
        type=float,
        required=True,
        help="per-call TTFT p95 SLO in seconds",
    )
    agentic.add_argument(
        "--itl-p95",
        type=float,
        required=True,
        help="per-call inter-token latency p95 SLO in seconds",
    )
    agentic.add_argument(
        "--task-p95",
        type=float,
        required=True,
        help="agent task end-to-end p95 SLO in seconds",
    )
    agentic.add_argument(
        "--deadline-ttft-slo",
        type=float,
        help="optional cumulative per-token deadline TTFT SLO in seconds (requires --deadline-tbt-slo)",
    )
    agentic.add_argument(
        "--deadline-tbt-slo",
        type=float,
        help="optional cumulative per-token deadline TBT SLO in seconds (requires --deadline-ttft-slo)",
    )
    agentic.add_argument(
        "--max-gap-slo",
        type=float,
        help="optional per-stream maximum content-event gap SLO in seconds",
    )
    agentic.add_argument(
        "--max-trace-events",
        type=int,
        default=DEFAULT_MAX_TRACE_EVENTS,
        help=f"maximum per-stream SSE content events recorded (default: {DEFAULT_MAX_TRACE_EVENTS})",
    )
    agentic.add_argument(
        "--cache-mode",
        choices=CACHE_MODES,
        default="unknown",
        help="operator-declared cache state, recorded and compared (default: unknown)",
    )
    agentic.add_argument(
        "--power-boundary",
        choices=POWER_BOUNDARIES,
        default="unspecified",
        help="what --watts measures (default: unspecified)",
    )
    agentic.add_argument(
        "--watts",
        type=float,
        help="measured power inside --power-boundary, in watts",
    )
    agentic.add_argument(
        "--power-method",
        help="how power was measured, e.g. wall meter model and sample interval",
    )
    agentic.add_argument("--request-timeout", type=float, default=600.0)
    agentic.add_argument(
        "--source-revision",
        help="repository revision recorded in the result; auto-detected when omitted",
    )
    agentic.add_argument("--note", help="free-form context recorded in the result")
    agentic.add_argument(
        "--output",
        default="-",
        help="result path, or - for stdout (default: -)",
    )

    interference = subparsers.add_parser(
        "interference",
        help="measure bounded decode victim streams under a frozen long-prompt prefill burst vs control",
    )
    interference.add_argument("--label", required=True, help="short target name")
    interference.add_argument("--base-url", required=True, help="http://host:port")
    interference.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served model name; discovered from /v1/models when omitted",
    )
    interference.add_argument(
        "--api-key",
        default=os.environ.get("QWEN38_BENCH_API_KEY"),
        help="bearer token, never written to the result (default: QWEN38_BENCH_API_KEY)",
    )
    interference.add_argument(
        "--header",
        action="append",
        help="additional request header as NAME: VALUE (repeatable)",
    )
    interference.add_argument(
        "--workload",
        default=str(DEFAULT_INTERFERENCE_WORKLOAD),
        help="frozen prefill-interference workload JSON (default: packaged fixture)",
    )
    interference.add_argument(
        "--control-mode",
        choices=CONTROL_MODES,
        default="short_prompt",
        help="matched control traffic mode (default: short_prompt)",
    )
    interference.add_argument("--runs", type=int, default=2)
    interference.add_argument("--warmup-runs", type=int, default=1)
    interference.add_argument("--settle-seconds", type=float, default=1.0)
    interference.add_argument(
        "--schedule-scale",
        type=float,
        default=1.0,
        help="uniformly scale dispatch offsets; recorded in the workload digest",
    )
    interference.add_argument(
        "--ttft-slo",
        type=float,
        required=True,
        help="TTFT SLO in seconds for victim streams",
    )
    interference.add_argument(
        "--tbt-slo",
        type=float,
        required=True,
        help="TBT SLO in seconds for cumulative token deadlines on victim streams",
    )
    interference.add_argument(
        "--max-gap-slo",
        type=float,
        help="optional distinct per-stream max content-event gap SLO in seconds (default: --tbt-slo)",
    )
    interference.add_argument(
        "--max-trace-events",
        type=int,
        default=DEFAULT_MAX_TRACE_EVENTS,
        help=f"maximum per-stream SSE content events recorded (default: {DEFAULT_MAX_TRACE_EVENTS})",
    )
    interference.add_argument(
        "--cache-mode",
        choices=CACHE_MODES,
        default="unknown",
        help="operator-declared cache state (default: unknown)",
    )
    interference.add_argument("--request-timeout", type=float, default=600.0)
    interference.add_argument(
        "--source-revision",
        help="repository revision recorded in the result; auto-detected when omitted",
    )
    interference.add_argument("--note", help="free-form context recorded in the result")
    interference.add_argument(
        "--output",
        default="-",
        help="result path, or - for stdout (default: -)",
    )

    compare = subparsers.add_parser("compare", help="compare result files")
    compare.add_argument("results", nargs="+", help="result JSON paths")
    compare.add_argument(
        "--baseline-label", help="target label to use as baseline (default: first)"
    )
    compare.add_argument(
        "--output",
        default="-",
        help="comparison path, or - for stdout (default: -)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.self_test:
            self_test()
            print("self-test: PASS")
            return 0
        if arguments.command == "run":
            if arguments.max_tokens <= 0:
                raise BenchmarkError("--max-tokens must be positive")
            if arguments.runs <= 0:
                raise BenchmarkError("--runs must be positive")
            if arguments.warmup_runs < 0:
                raise BenchmarkError("--warmup-runs must be non-negative")
            if (
                not math.isfinite(arguments.settle_seconds)
                or arguments.settle_seconds < 0
            ):
                raise BenchmarkError("--settle-seconds must be finite and non-negative")
            if (
                not math.isfinite(arguments.request_timeout)
                or arguments.request_timeout <= 0
            ):
                raise BenchmarkError("--request-timeout must be finite and positive")
            deadline_slo = _validate_deadline_slo({
                "ttft_slo_seconds": arguments.ttft_slo,
                "tbt_slo_seconds": arguments.tbt_slo,
                "max_gap_slo_seconds": arguments.max_gap_slo,
            })
            widths = parse_concurrency(arguments.concurrency)
            corpus = load_corpus(Path(arguments.corpus))
            endpoint = Endpoint(
                label=arguments.label,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                headers=parse_headers(arguments.header),
                timeout=arguments.request_timeout,
            )
            result = run_benchmark(
                endpoint,
                corpus,
                max_tokens=arguments.max_tokens,
                concurrency=widths,
                runs=arguments.runs,
                warmup_runs=arguments.warmup_runs,
                stream=not arguments.no_stream,
                settle_seconds=arguments.settle_seconds,
                revision=arguments.source_revision,
                note=arguments.note,
                max_trace_events=arguments.max_trace_events,
                deadline_slo=deadline_slo,
            )
            write_json(result, arguments.output)
            return 0
        if arguments.command == "agentic":
            if arguments.runs <= 0:
                raise BenchmarkError("--runs must be positive")
            if arguments.warmup_runs < 0:
                raise BenchmarkError("--warmup-runs must be non-negative")
            if (
                not math.isfinite(arguments.settle_seconds)
                or arguments.settle_seconds < 0
            ):
                raise BenchmarkError("--settle-seconds must be finite and non-negative")
            if (
                not math.isfinite(arguments.tool_wait_scale)
                or arguments.tool_wait_scale < 0
            ):
                raise BenchmarkError(
                    "--tool-wait-scale must be finite and non-negative"
                )
            if (
                not math.isfinite(arguments.request_timeout)
                or arguments.request_timeout <= 0
            ):
                raise BenchmarkError("--request-timeout must be finite and positive")
            slo = {
                "ttft_p95_seconds": arguments.ttft_p95,
                "itl_p95_seconds": arguments.itl_p95,
                "task_p95_seconds": arguments.task_p95,
            }
            for name, value in slo.items():
                if not math.isfinite(value) or value <= 0:
                    raise BenchmarkError(f"SLO {name} must be finite and positive")
            deadline_slo = _validate_deadline_slo({
                "ttft_slo_seconds": arguments.deadline_ttft_slo,
                "tbt_slo_seconds": arguments.deadline_tbt_slo,
                "max_gap_slo_seconds": arguments.max_gap_slo,
            })
            widths = parse_agent_widths(arguments.agents)
            trace = load_agentic_trace(Path(arguments.trace))
            endpoint = Endpoint(
                label=arguments.label,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                headers=parse_headers(arguments.header),
                timeout=arguments.request_timeout,
            )
            result = run_agentic_benchmark(
                endpoint,
                trace,
                agents=widths,
                runs=arguments.runs,
                warmup_runs=arguments.warmup_runs,
                settle_seconds=arguments.settle_seconds,
                tool_wait_scale=arguments.tool_wait_scale,
                slo=slo,
                cache_mode=arguments.cache_mode,
                power={
                    "boundary": arguments.power_boundary,
                    "watts": arguments.watts,
                    "method": arguments.power_method,
                },
                revision=arguments.source_revision,
                note=arguments.note,
                deadline_slo=deadline_slo,
                max_trace_events=arguments.max_trace_events,
            )
            validate_agentic_result(result, "agentic benchmark")
            write_json(result, arguments.output)
            return 0 if result["summary"]["run_status"] != "failed_population" else 1
        if arguments.command == "interference":
            if arguments.runs <= 0:
                raise BenchmarkError("--runs must be positive")
            if arguments.warmup_runs < 0:
                raise BenchmarkError("--warmup-runs must be non-negative")
            if (
                not math.isfinite(arguments.settle_seconds)
                or arguments.settle_seconds < 0
            ):
                raise BenchmarkError("--settle-seconds must be finite and non-negative")
            if (
                not math.isfinite(arguments.schedule_scale)
                or arguments.schedule_scale < 0
            ):
                raise BenchmarkError("--schedule-scale must be finite and non-negative")
            if (
                not math.isfinite(arguments.request_timeout)
                or arguments.request_timeout <= 0
            ):
                raise BenchmarkError("--request-timeout must be finite and positive")
            deadline_slo = _validate_deadline_slo({
                "ttft_slo_seconds": arguments.ttft_slo,
                "tbt_slo_seconds": arguments.tbt_slo,
                "max_gap_slo_seconds": (
                    arguments.max_gap_slo
                    if arguments.max_gap_slo is not None
                    else arguments.tbt_slo
                ),
            })
            fixture = load_interference_workload(Path(arguments.workload))
            endpoint = Endpoint(
                label=arguments.label,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                headers=parse_headers(arguments.header),
                timeout=arguments.request_timeout,
            )
            result = run_prefill_interference_benchmark(
                endpoint,
                fixture,
                control_mode=arguments.control_mode,
                runs=arguments.runs,
                warmup_runs=arguments.warmup_runs,
                settle_seconds=arguments.settle_seconds,
                schedule_scale=arguments.schedule_scale,
                deadline_slo=deadline_slo,
                cache_mode=arguments.cache_mode,
                max_trace_events=arguments.max_trace_events,
                revision=arguments.source_revision,
                note=arguments.note,
            )
            validate_interference_result(result, "interference benchmark")
            write_json(result, arguments.output)
            return 0 if result["summary"]["run_status"] == "passed" else 1
        if arguments.command == "compare":
            results = [load_result(Path(path)) for path in arguments.results]
            comparison = compare_results(results, arguments.baseline_label)
            write_json(comparison, arguments.output)
            return 0 if comparison["text_parity"]["matched"] else 1
        parser.error("a subcommand is required (run, agentic, interference, or compare)")
    except BenchmarkError as error:
        print(f"qwen38-benchmark: {error}", file=sys.stderr)
        return 2
    except (OSError, http.client.HTTPException) as error:
        print(
            f"qwen38-benchmark: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
