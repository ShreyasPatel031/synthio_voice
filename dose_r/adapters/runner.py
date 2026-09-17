"""Runs the DOSE set through a configured adapter.

Resumability is the point: a 274-item run against a paid API that dies at item
200 must not re-buy those 200 items. An item counts as done only if its manifest
record says ok, its audio file is still there, and the text it was synthesised
from hashes to the text in the dataset today -- so a dataset fix invalidates
exactly the rows it touched and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .base import SynthesisRequest, SynthesisRecord, TTSAdapter, sha256_hex
from .config import DEFAULT_SYSTEMS_PATH, SystemConfig, load_system
from .registry import build_adapter

DEFAULT_DATASET = Path("data/dose_v1.jsonl")
ITEM_FIELDS = ("name", "name_type", "is_combination", "ingredients", "spans")


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def to_request(row: dict[str, Any]) -> SynthesisRequest:
    return SynthesisRequest(
        item_id=row["id"],
        text=row["sentence"],
        item={k: row[k] for k in ITEM_FIELDS if k in row},
    )


@dataclass
class RunPaths:
    root: Path

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def meta(self) -> Path:
        return self.root / "run.json"

    def prepare(self) -> "RunPaths":
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        return self


class ManifestWriter:
    """Append-only JSONL, fsynced per line. A killed run leaves a manifest that
    is still valid up to its last complete line."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Last record per item wins, so a retried item supersedes its failure."""
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from a killed run
        records[record["item_id"]] = record
    return records


def completed_items(
    manifest: dict[str, dict[str, Any]], rows: Iterable[dict[str, Any]], paths: RunPaths, system_id: str
) -> set[str]:
    by_id = {row["id"]: row for row in rows}
    done = set()
    for item_id, record in manifest.items():
        row = by_id.get(item_id)
        if row is None or record.get("status") != "ok":
            continue
        if record.get("system_id") != system_id:
            continue
        if record.get("text_sha256") != sha256_hex(row["sentence"]):
            continue
        audio_path = record.get("audio_path")
        if not audio_path or not (paths.audio_dir / Path(audio_path).name).exists():
            continue
        done.add(item_id)
    return done


@dataclass
class RunResult:
    run_id: str
    paths: RunPaths
    system: SystemConfig
    skipped: list[str]
    records: list[SynthesisRecord]
    summary: dict[str, Any]

    @property
    def failures(self) -> list[SynthesisRecord]:
        return [r for r in self.records if not r.ok]


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": round(ordered[-1], 2),
    }


def summarize(
    records: list[SynthesisRecord], skipped: list[str], manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    prior_ok = [manifest[i] for i in skipped if i in manifest]

    errors: dict[str, int] = {}
    for record in failed:
        errors[record.error_class or "Unknown"] = errors.get(record.error_class or "Unknown", 0) + 1

    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    ttfas = [r.ttfa_ms for r in ok if r.ttfa_ms is not None]
    retried = sum(1 for r in records if len(r.attempts) > 1)

    return {
        "synthesized": len(ok),
        "failed": len(failed),
        "resumed_skipped": len(skipped),
        "items_with_retries": retried,
        "errors_by_class": errors,
        "cost_usd_this_run": round(sum(r.cost_usd for r in ok), 6),
        "cost_usd_including_resumed": round(
            sum(r.cost_usd for r in ok) + sum(float(p.get("cost_usd") or 0) for p in prior_ok), 6
        ),
        "audio_seconds": round(sum(r.usage.audio_seconds for r in ok), 2),
        "latency_ms": _percentiles(latencies),
        "ttfa_ms": _percentiles(ttfas),
        "ttfa_coverage": f"{len(ttfas)}/{len(ok)}",
    }


def run(
    system: SystemConfig,
    rows: list[dict[str, Any]],
    run_dir: str | Path,
    adapter: TTSAdapter | None = None,
    resume: bool = True,
    concurrency: int | None = None,
    dataset_path: str | Path = DEFAULT_DATASET,
    on_record: Callable[[SynthesisRecord], None] | None = None,
) -> RunResult:
    paths = RunPaths(Path(run_dir)).prepare()
    run_id = paths.root.name
    manifest = load_manifest(paths.manifest) if resume else {}
    skipped = sorted(completed_items(manifest, rows, paths, system.system_id)) if resume else []
    pending = [row for row in rows if row["id"] not in set(skipped)]

    writer = ManifestWriter(paths.manifest)
    owned = adapter is None
    adapter = adapter or build_adapter(system)
    workers = concurrency or system.concurrency
    records: list[SynthesisRecord] = []
    records_lock = threading.Lock()

    def work(row: dict[str, Any]) -> SynthesisRecord:
        record = adapter.synthesize(to_request(row), run_id=run_id)
        if record.ok and record.audio is not None:
            filename = f"{record.item_id}.{record.audio_format.extension}"
            (paths.audio_dir / filename).write_bytes(record.audio)
            record.audio_path = str(paths.audio_dir / filename)
        writer.append(record.to_dict())
        with records_lock:
            records.append(record)
        if on_record:
            on_record(record)
        return record

    try:
        adapter.preflight()
        if pending:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                list(pool.map(work, pending))
    finally:
        if owned:
            adapter.close()

    records.sort(key=lambda r: r.item_id)
    summary = summarize(records, skipped, manifest)
    meta = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path),
            "rows": len(rows),
            "ingredient_spans": sum(len(r.get("ingredients") or []) for r in rows),
            "sha256": sha256_hex(Path(dataset_path).read_bytes())
            if Path(dataset_path).exists()
            else None,
        },
        "system": system.to_dict(),
        "concurrency": workers,
        "resume": resume,
        "summary": summary,
    }
    paths.meta.write_text(json.dumps(meta, indent=2) + "\n")
    return RunResult(run_id, paths, system, skipped, records, summary)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the DOSE set through a TTS adapter.")
    ap.add_argument("--system", default="mock")
    ap.add_argument("--systems-file", default=str(DEFAULT_SYSTEMS_PATH))
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--run-dir", default=None, help="default: runs/<system>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    system = load_system(args.system, args.systems_file)
    if system.status != "ready":
        print(f"system {system.system_id!r} is status={system.status}", file=sys.stderr)
        return 2

    rows = load_dataset(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    run_dir = args.run_dir or Path("runs") / system.system_id
    done = 0
    total = len(rows)

    def progress(record: SynthesisRecord) -> None:
        nonlocal done
        done += 1
        mark = "ok " if record.ok else "ERR"
        print(f"[{done}/{total}] {mark} {record.item_id} {record.latency_ms or 0:.0f}ms", file=sys.stderr)

    result = run(
        system,
        rows,
        run_dir,
        resume=not args.no_resume,
        concurrency=args.concurrency,
        dataset_path=args.dataset,
        on_record=progress,
    )
    print(json.dumps({"run_dir": str(result.paths.root), **result.summary}, indent=2))
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
