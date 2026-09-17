"""Benchmark runner: drives adapters over the DOSE items and logs everything.

Design notes:

* Every call's audio, wall-clock latency and estimated cost are persisted next to
  the score. That log is the raw material for both the replication-fidelity report
  (1d) and the accuracy/latency/cost tradeoff blueprint (2a), so it is written even
  when scoring is deferred.
* Runs are resumable. A 274-item sweep across several systems is long enough that
  losing it to one transient failure is a real cost.
* Concurrency is per-system, so one slow backend does not serialise the others,
  but latency figures stay comparable because each system's calls share a worker
  pool of fixed size.
"""

from __future__ import annotations

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from . import __version__, config
from .adapters import SynthesisResult, build_adapter
from .dataset import DoseItem
from .scoring import Scorer


@dataclass
class RunConfig:
    systems: list[str]
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    )
    concurrency: int = 4
    max_attempts: int = 3
    save_audio: bool = True
    limit: int | None = None
    notes: str = ""


class BenchmarkRunner:
    def __init__(self, cfg: RunConfig, scorer: Scorer | None = None,
                 runs_dir: Path | None = None):
        self.cfg = cfg
        self.scorer = scorer
        self.run_dir = (runs_dir or config.RUNS_DIR) / cfg.run_id
        self.audio_dir = self.run_dir / "audio"
        self.results_path = self.run_dir / "results.jsonl"

    # --- persistence -----------------------------------------------------
    def _completed_keys(self) -> set[tuple[str, str]]:
        """(system_id, item_id) pairs already logged, for resume."""
        if not self.results_path.exists():
            return set()
        done = set()
        with self.results_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from an interrupted run
                if rec.get("synthesis", {}).get("ok"):
                    done.add((rec["system_id"], rec["item_id"]))
        return done

    def _audio_path(self, system_id: str, item_id: str, fmt: str) -> Path:
        return self.audio_dir / system_id / f"{item_id}.{fmt}"

    def _write_manifest(self, items: list[DoseItem], extra: dict) -> None:
        manifest = {
            "run_id": self.cfg.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dose_r_version": __version__,
            "systems": self.cfg.systems,
            "scorer": self.scorer.scorer_id if self.scorer else None,
            "scorer_measures_pronunciation": (
                self.scorer.measures_pronunciation if self.scorer else None
            ),
            "n_items": len(items),
            "concurrency": self.cfg.concurrency,
            "max_attempts": self.cfg.max_attempts,
            "gcp_project": config.GCP_PROJECT,
            "platform": platform.platform(),
            "notes": self.cfg.notes,
            **extra,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- execution -------------------------------------------------------
    def run(self, items: Iterable[DoseItem],
            progress: Callable[[str], None] | None = None) -> Path:
        items = list(items)
        if self.cfg.limit:
            items = items[: self.cfg.limit]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(items, {"status": "running"})

        already = self._completed_keys()
        if already and progress:
            progress(f"resuming: {len(already)} calls already logged")

        t_start = time.perf_counter()
        totals: dict[str, dict] = {}

        with self.results_path.open("a") as sink:
            for system_id in self.cfg.systems:
                adapter = build_adapter(system_id)
                todo = [i for i in items if (system_id, i.item_id) not in already]
                if progress:
                    progress(f"{system_id}: {len(todo)} to synthesize "
                             f"({len(items) - len(todo)} cached)")

                agg = {"ok": 0, "failed": 0, "cost_usd": 0.0, "latencies": []}

                with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
                    futures = {
                        pool.submit(
                            adapter.synthesize, item.sentence, item.item_id,
                            max_attempts=self.cfg.max_attempts,
                        ): item
                        for item in todo
                    }
                    for n, fut in enumerate(as_completed(futures), start=1):
                        item = futures[fut]
                        try:
                            res: SynthesisResult = fut.result()
                        except Exception as exc:
                            res = SynthesisResult(
                                system_id=system_id, item_id=item.item_id, ok=False,
                                error=f"runner: {type(exc).__name__}: {exc}",
                            )

                        rec = self._persist(item, res)
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()

                        if res.ok:
                            agg["ok"] += 1
                            agg["cost_usd"] += res.cost_usd or 0.0
                            if res.total_ms:
                                agg["latencies"].append(res.total_ms)
                        else:
                            agg["failed"] += 1

                        if progress and (n % 25 == 0 or n == len(todo)):
                            progress(f"  {system_id}: {n}/{len(todo)}")

                totals[system_id] = {
                    "ok": agg["ok"], "failed": agg["failed"],
                    "cost_usd": round(agg["cost_usd"], 6),
                }

        self._write_manifest(items, {
            "status": "complete",
            "wall_clock_s": round(time.perf_counter() - t_start, 2),
            "per_system": totals,
        })
        return self.results_path

    def _persist(self, item: DoseItem, res: SynthesisResult) -> dict:
        """Write audio to disk and build the JSONL record."""
        audio_rel = None
        if self.cfg.save_audio and res.ok and res.audio:
            path = self._audio_path(res.system_id, item.item_id, res.audio_format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(res.audio)
            audio_rel = str(path.relative_to(self.run_dir))

        rec = {
            "run_id": self.cfg.run_id,
            "system_id": res.system_id,
            "item_id": item.item_id,
            "drug": item.drug,
            "drug_raw": item.drug_raw,
            "name_type": item.name_type,
            "sentence": item.sentence,
            "audio_path": audio_rel,
            "synthesis": res.to_record(),
            "score": None,
        }

        if self.scorer is not None:
            rec["score"] = self.scorer.score(item, res).to_record()

        return rec
