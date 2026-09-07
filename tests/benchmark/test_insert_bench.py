"""Client-only benchmarks for row-insert request construction."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pytest
from google.protobuf.internal import api_implementation
from pymilvus import DataType
from pymilvus.client import entity_helper
from pymilvus.client.prepare import Prepare
from pymilvus.grpc_gen import milvus_pb2, schema_pb2

_MIB = 1024 * 1024
_RSS_SAMPLE_INTERVAL_SECONDS = 0.001
# The returned request should stay close to one wire payload, retaining the serialized bytes
# should stay close to two, and upb's transient serialization work should stay near four.
_PACKED_RSS_LIMIT_RATIO = 1.25
_SERIALIZED_RSS_LIMIT_RATIO = 2.25
_LIFECYCLE_PEAK_RSS_LIMIT_RATIO = 4.0
# Cover allocator/page granularity for the smaller representative batch.
_RSS_ASSERTION_SLACK_BYTES = 4 * _MIB
# Allow sampling jitter while still rejecting a material regression from the legacy path.
_LEGACY_COMPARISON_SLACK_BYTES = 2 * _MIB
_COMPARISON_ROUNDS = 3


def _current_rss_bytes(pid: int) -> int:
    statm = (Path("/proc") / str(pid) / "statm").read_text(encoding="ascii")
    resident_pages = int(statm.split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _write_worker_command(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(f"{command}\n")
    process.stdin.flush()


def _read_worker_event(process: subprocess.Popen[str], expected_phase: str) -> dict[str, Any]:
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        assert process.stderr is not None
        stderr = process.stderr.read()
        raise RuntimeError(f"row-insert memory worker exited before {expected_phase}: {stderr}")
    event = json.loads(line)
    if event.get("phase") != expected_phase:
        raise RuntimeError(f"expected worker phase {expected_phase}, got {event}")
    return event


def _emit_worker_event(event: dict[str, Any]) -> None:
    sys.stdout.write(f"{json.dumps(event)}\n")
    sys.stdout.flush()


def _measure_row_insert_memory(row_count: int, dim: int, strategy: str) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(row_count),
            str(dim),
            strategy,
        ],
        cwd=repository_root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    sampling = threading.Event()
    sampler: threading.Thread | None = None
    peak_rss = [0]
    try:
        ready = _read_worker_event(process, "ready")
        baseline_rss = _current_rss_bytes(process.pid)
        peak_rss[0] = baseline_rss

        def sample_rss() -> None:
            while sampling.is_set():
                try:
                    peak_rss[0] = max(peak_rss[0], _current_rss_bytes(process.pid))
                except FileNotFoundError:
                    break
                time.sleep(_RSS_SAMPLE_INTERVAL_SECONDS)

        sampling.set()
        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()

        _write_worker_command(process, "pack")
        packed = _read_worker_event(process, "packed")
        packed_rss = _current_rss_bytes(process.pid)
        packing_peak_rss = max(peak_rss[0], packed_rss)

        _write_worker_command(process, "serialize")
        serialized = _read_worker_event(process, "serialized")
        serialized_rss = _current_rss_bytes(process.pid)
        lifecycle_peak_rss = max(peak_rss[0], serialized_rss)

        sampling.clear()
        sampler.join(timeout=1)
        _write_worker_command(process, "stop")
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            raise RuntimeError(f"row-insert memory worker failed: {stderr}{stdout}")

        return {
            "raw_bytes": ready["raw_bytes"],
            "wire_bytes": serialized["wire_bytes"],
            "packing_seconds": packed["packing_seconds"],
            "serialization_seconds": serialized["serialization_seconds"],
            "packed_rss_delta_bytes": packed_rss - baseline_rss,
            "serialized_rss_delta_bytes": serialized_rss - baseline_rss,
            "packing_peak_rss_delta_bytes": packing_peak_rss - baseline_rss,
            "lifecycle_peak_rss_delta_bytes": lifecycle_peak_rss - baseline_rss,
        }
    finally:
        sampling.clear()
        if sampler is not None:
            sampler.join(timeout=1)
        if process.poll() is None:
            process.kill()
            process.communicate()


def _compare_row_insert_memory(row_count: int, dim: int) -> dict[str, dict[str, Any]]:
    samples = {"legacy": [], "current": []}
    for _ in range(_COMPARISON_ROUNDS):
        for strategy, strategy_samples in samples.items():
            strategy_samples.append(_measure_row_insert_memory(row_count, dim, strategy))

    return {
        strategy: {
            metric: median(sample[metric] for sample in strategy_samples)
            for metric in strategy_samples[0]
        }
        for strategy, strategy_samples in samples.items()
    }


@pytest.mark.skipif(sys.platform != "linux", reason="RSS sampling requires Linux /proc")
@pytest.mark.skipif(
    api_implementation.Type() != "upb", reason="memory bounds target protobuf's upb runtime"
)
@pytest.mark.parametrize("row_count", [1_000, 10_000])
def test_row_insert_packing_and_serialization_memory(benchmark, row_count: int) -> None:
    """Guard retained and peak RSS across the complete unary request lifecycle."""
    dim = 768
    result = benchmark.pedantic(
        _compare_row_insert_memory,
        args=(row_count, dim),
        iterations=1,
        rounds=1,
    )
    benchmark.extra_info.update(
        {
            f"{strategy}_{metric}": value
            for strategy, measurements in result.items()
            for metric, value in measurements.items()
        }
    )

    legacy = result["legacy"]
    current = result["current"]
    wire_bytes = current["wire_bytes"]
    assert wire_bytes == legacy["wire_bytes"]
    assert wire_bytes >= current["raw_bytes"]
    assert (
        current["packed_rss_delta_bytes"]
        <= _PACKED_RSS_LIMIT_RATIO * wire_bytes + _RSS_ASSERTION_SLACK_BYTES
    )
    assert (
        current["serialized_rss_delta_bytes"]
        <= _SERIALIZED_RSS_LIMIT_RATIO * wire_bytes + _RSS_ASSERTION_SLACK_BYTES
    )
    assert (
        current["lifecycle_peak_rss_delta_bytes"]
        <= _LIFECYCLE_PEAK_RSS_LIMIT_RATIO * wire_bytes + _RSS_ASSERTION_SLACK_BYTES
    )
    for metric in (
        "packed_rss_delta_bytes",
        "serialized_rss_delta_bytes",
        "lifecycle_peak_rss_delta_bytes",
    ):
        assert current[metric] <= legacy[metric] + _LEGACY_COMPARISON_SLACK_BYTES


def _legacy_row_insert_request(rows, fields_info):
    """Reproduce the merge-base standalone FieldData strategy for this workload."""
    request = milvus_pb2.InsertRequest(
        collection_name="insert_memory_bench", partition_name="", num_rows=len(rows)
    )
    field_info = fields_info[1]
    field_data = schema_pb2.FieldData(field_name=field_info["name"], type=field_info["type"])
    vector_bytes_cache = {}
    for row in rows:
        entity_helper.pack_field_value_to_field_data(
            row[field_info["name"]], field_data, field_info, vector_bytes_cache
        )
    entity_helper.flush_vector_bytes(field_data, vector_bytes_cache)
    request.fields_data.extend([field_data])
    return request


def _row_insert_memory_worker(row_count: int, dim: int) -> None:
    vectors = np.random.default_rng(0).random((row_count, dim), dtype=np.float32)
    rows = [{"vector": vector} for vector in vectors]
    fields_info = [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
        {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}},
    ]
    gc.collect()
    _emit_worker_event({"phase": "ready", "raw_bytes": vectors.nbytes})

    if input() != "pack":
        raise RuntimeError("memory worker expected pack command")
    started = time.perf_counter()
    strategy = sys.argv[4]
    if strategy == "legacy":
        request = _legacy_row_insert_request(rows, fields_info)
    elif strategy == "current":
        request = Prepare.row_insert_param("insert_memory_bench", rows, "", fields_info=fields_info)
    else:
        raise RuntimeError(f"unknown memory worker strategy: {strategy}")
    _emit_worker_event({"phase": "packed", "packing_seconds": time.perf_counter() - started})

    if input() != "serialize":
        raise RuntimeError("memory worker expected serialize command")
    started = time.perf_counter()
    wire = request.SerializeToString()
    _emit_worker_event(
        {
            "phase": "serialized",
            "serialization_seconds": time.perf_counter() - started,
            "wire_bytes": len(wire),
        }
    )

    if input() != "stop":
        raise RuntimeError("memory worker expected stop command")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_insert_bench.py --worker ROW_COUNT DIM STRATEGY")
    _row_insert_memory_worker(int(sys.argv[2]), int(sys.argv[3]))
