from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
import torch.nn.functional as F

from app.config import Settings
from app.storage import SpeakerRecord, SpeakerStore

LOGGER = logging.getLogger(__name__)
ALLOWED_AUDIO_SUFFIXES = {".wav", ".opus", ".ogg", ".flac", ".mp3", ".m4a"}


class ServiceError(RuntimeError):
    pass


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    dimension: int
    duration_seconds: float


class SpeakerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.model_device)
        self.store = SpeakerStore(settings.store_path)
        self.model = None
        self.embedding_dim = 0

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise ServiceError("CUDA was requested but is not available inside the container.")
        return torch.device("cpu")

    def load(self) -> None:
        os.environ.setdefault("HF_HOME", self.settings.hf_home)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", str(self.settings.hf_hub_disable_telemetry))
        if self.settings.hf_token:
            os.environ.setdefault("HF_TOKEN", self.settings.hf_token)

        LOGGER.info("Loading speaker model %s on %s", self.settings.model_id, self.device)
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.settings.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        warmup = self._warmup()
        self.embedding_dim = warmup.dimension
        LOGGER.info("Model ready with embedding dimension %s", self.embedding_dim)

    def _warmup(self) -> EmbeddingResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "warmup.wav"
            self._write_silence(wav_path)
            return self.embedding_from_path(wav_path)

    @staticmethod
    def _write_silence(path: Path, sample_rate: int = 16000, seconds: int = 1) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * sample_rate * seconds)

    def embedding_from_upload(self, filename: str, payload: bytes) -> EmbeddingResult:
        if len(payload) > self.settings.max_upload_bytes:
            raise ServiceError(f"Upload exceeds MAX_UPLOAD_BYTES={self.settings.max_upload_bytes}.")
        suffix = Path(filename or "upload").suffix.lower() or ".bin"
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / f"upload{suffix}"
            output_path = Path(tmpdir) / "normalized.wav"
            input_path.write_bytes(payload)
            self._normalize_audio(input_path, output_path)
            return self.embedding_from_path(output_path)

    def embedding_from_path(self, audio_path: Path) -> EmbeddingResult:
        assert self.model is not None
        with torch.inference_mode():
            embedding = self.model.get_embedding(str(audio_path))
        vector = embedding.detach().to("cpu").squeeze(0).numpy().astype(np.float32)
        vector = self._normalize_vector(vector)
        duration_seconds = self._probe_duration(audio_path)
        return EmbeddingResult(vector=vector, dimension=int(vector.shape[0]), duration_seconds=duration_seconds)

    def enroll_archive(self, label: str, archive_bytes: bytes) -> SpeakerRecord:
        if not label.strip():
            raise ServiceError("Label must not be empty.")
        embeddings: list[np.ndarray] = []
        source_files: list[str] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive, tempfile.TemporaryDirectory() as tmpdir:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.lower() in ALLOWED_AUDIO_SUFFIXES
            ]
            if not members:
                raise ServiceError("Archive did not contain supported audio files.")
            if len(members) > self.settings.max_enroll_files:
                raise ServiceError(
                    f"Archive contains {len(members)} files which exceeds MAX_ENROLL_FILES={self.settings.max_enroll_files}."
                )
            for member in sorted(members, key=lambda item: item.filename):
                raw_name = Path(member.filename).name
                suffix = Path(raw_name).suffix.lower()
                input_path = Path(tmpdir) / raw_name
                output_path = Path(tmpdir) / f"{Path(raw_name).stem}.wav"
                input_path.write_bytes(archive.read(member))
                self._normalize_audio(input_path, output_path)
                embeddings.append(self.embedding_from_path(output_path).vector)
                source_files.append(raw_name)

        centroid = self._normalize_vector(np.mean(np.stack(embeddings), axis=0))
        record = SpeakerRecord(
            label=label.strip(),
            centroid=centroid,
            embedding_count=len(embeddings),
            source_files=source_files,
        )
        self.store.upsert(record)
        return record

    def verify_against_label(self, label: str, filename: str, payload: bytes) -> tuple[SpeakerRecord, float, bool]:
        record = self.store.get(label)
        if record is None:
            raise ServiceError(f"Speaker '{label}' is not enrolled.")
        probe = self.embedding_from_upload(filename, payload)
        score = self.cosine_similarity(probe.vector, record.centroid)
        return record, score, bool(score >= self.settings.verification_threshold)

    def verify_pair(self, left_name: str, left_payload: bytes, right_name: str, right_payload: bytes) -> tuple[float, bool]:
        left = self.embedding_from_upload(left_name, left_payload)
        right = self.embedding_from_upload(right_name, right_payload)
        score = self.cosine_similarity(left.vector, right.vector)
        return score, bool(score >= self.settings.verification_threshold)

    def identify(self, filename: str, payload: bytes) -> dict[str, object]:
        candidates = self.store.list()
        if not candidates:
            raise ServiceError("No enrolled speakers are available for identification.")
        probe = self.embedding_from_upload(filename, payload)
        scores = [
            {"label": record.label, "score": self.cosine_similarity(probe.vector, record.centroid)}
            for record in candidates
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        best = scores[0]
        accepted = bool(best["score"] >= self.settings.identification_threshold)
        return {
            "scores": scores,
            "best_score": best["score"],
            "predicted_label": best["label"] if accepted else "unknown",
            "unknown": not accepted,
            "embedding_dimension": probe.dimension,
            "duration_seconds": probe.duration_seconds,
        }

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_tensor = torch.from_numpy(left)
        right_tensor = torch.from_numpy(right)
        score = F.cosine_similarity(left_tensor.unsqueeze(0), right_tensor.unsqueeze(0)).item()
        return float(score)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ServiceError("Encountered a zero-norm embedding.")
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _probe_duration(audio_path: Path) -> float:
        with wave.open(str(audio_path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 4)

    @staticmethod
    def _normalize_audio(input_path: Path, output_path: Path) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ServiceError(result.stderr.strip() or f"Failed to normalize {input_path.name}.")
