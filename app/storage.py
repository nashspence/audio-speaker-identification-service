from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SpeakerRecord:
    label: str
    centroid: np.ndarray
    embedding_count: int
    source_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "centroid": self.centroid.tolist(),
            "embedding_count": self.embedding_count,
            "source_files": self.source_files,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerRecord":
        return cls(
            label=payload["label"],
            centroid=np.asarray(payload["centroid"], dtype=np.float32),
            embedding_count=int(payload["embedding_count"]),
            source_files=list(payload.get("source_files", [])),
        )


class SpeakerStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, SpeakerRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = {
            label: SpeakerRecord.from_dict(record)
            for label, record in payload.get("speakers", {}).items()
        }

    def _flush(self) -> None:
        payload = {
            "speakers": {label: record.to_dict() for label, record in sorted(self._records.items())}
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert(self, record: SpeakerRecord) -> None:
        self._records[record.label] = record
        self._flush()

    def get(self, label: str) -> SpeakerRecord | None:
        return self._records.get(label)

    def list(self) -> list[SpeakerRecord]:
        return [self._records[label] for label in sorted(self._records)]

    def count(self) -> int:
        return len(self._records)
