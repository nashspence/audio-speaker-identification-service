from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.service import ServiceError, SpeakerService

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = SpeakerService(settings)
    service.load()
    app.state.speaker_service = service
    yield


app = FastAPI(
    title="TitaNet Speaker ID Service",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


def get_service() -> SpeakerService:
    return app.state.speaker_service


def read_upload(upload: UploadFile) -> tuple[str, bytes]:
    payload = upload.file.read()
    return upload.filename or "upload.bin", payload


@app.exception_handler(ServiceError)
async def service_error_handler(_, exc: ServiceError):
    return ORJSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/healthz")
def healthz():
    service = get_service()
    return {
        "status": "ok",
        "ready": True,
        "model": settings.model_id,
        "device": str(service.device),
        "embedding_dimension": service.embedding_dim,
        "enrolled_speakers": service.store.count(),
    }


@app.post("/v1/embeddings")
def extract_embedding(file: UploadFile = File(...)):
    service = get_service()
    filename, payload = read_upload(file)
    result = service.embedding_from_upload(filename, payload)
    return {
        "object": "speaker_embedding",
        "model": settings.model_id,
        "device": str(service.device),
        "dimension": result.dimension,
        "duration_seconds": result.duration_seconds,
        "embedding": result.vector.tolist(),
    }


@app.post("/v1/enroll")
def enroll_speaker(label: str = Form(...), archive: UploadFile = File(...)):
    service = get_service()
    filename, payload = read_upload(archive)
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Enrollment archive must be a .zip file.")
    record = service.enroll_archive(label=label, archive_bytes=payload)
    return {
        "object": "speaker_enrollment",
        "model": settings.model_id,
        "speaker": {
            "label": record.label,
            "embedding_count": record.embedding_count,
            "source_files": record.source_files,
            "centroid_dimension": int(record.centroid.shape[0]),
        },
        "enrolled_speakers": service.store.count(),
    }


@app.post("/v1/verify")
def verify_speaker(
    label: str | None = Form(default=None),
    file: UploadFile = File(...),
    reference: UploadFile | None = File(default=None),
):
    service = get_service()
    probe_name, probe_payload = read_upload(file)
    if label:
        record, score, match = service.verify_against_label(label=label, filename=probe_name, payload=probe_payload)
        return {
            "object": "speaker_verification",
            "mode": "enrolled_label",
            "model": settings.model_id,
            "speaker_label": record.label,
            "score": score,
            "threshold": settings.verification_threshold,
            "match": match,
        }
    if reference is None:
        raise HTTPException(status_code=400, detail="Provide either a speaker label or a reference file.")
    ref_name, ref_payload = read_upload(reference)
    score, match = service.verify_pair(
        left_name=probe_name,
        left_payload=probe_payload,
        right_name=ref_name,
        right_payload=ref_payload,
    )
    return {
        "object": "speaker_verification",
        "mode": "pairwise",
        "model": settings.model_id,
        "score": score,
        "threshold": settings.verification_threshold,
        "match": match,
    }


@app.post("/v1/identify")
def identify_speaker(file: UploadFile = File(...)):
    service = get_service()
    filename, payload = read_upload(file)
    result = service.identify(filename=filename, payload=payload)
    return {
        "object": "speaker_identification",
        "model": settings.model_id,
        "threshold": settings.identification_threshold,
        "predicted_label": result["predicted_label"],
        "unknown": result["unknown"],
        "best_score": result["best_score"],
        "scores": result["scores"],
        "embedding_dimension": result["embedding_dimension"],
        "duration_seconds": result["duration_seconds"],
    }
