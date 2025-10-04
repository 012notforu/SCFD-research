from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


def create_run_directory(tag: str, outdir: Optional[str] = None, root: str = "runs") -> Path:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(outdir) if outdir else Path(root) / f"{timestamp}_{tag}"
    base.mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(exist_ok=True)
    return base


@dataclass
class RunLogger:
    directory: Path

    def __post_init__(self) -> None:
        self.jsonl_path = self.directory / "logs.jsonl"
        self.csv_files: Dict[str, Path] = {}

    def log_step(self, payload: Dict[str, float]) -> None:
        payload = dict(payload)
        payload.setdefault("timestamp", time.time())
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def log_csv(self, name: str, headers: Iterable[str], row: Iterable[float]) -> None:
        path = self.csv_files.setdefault(name, self.directory / f"{name}.csv")
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(list(headers))
            writer.writerow(list(row))

    def dump_config(self, cfg_text: str) -> None:
        (self.directory / "cfg.yaml").write_text(cfg_text, encoding="utf-8")
