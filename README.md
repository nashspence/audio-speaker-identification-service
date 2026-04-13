# Audio Speaker Identification Service

Minimal GPU-backed HTTP service for `nvidia/speakerverification_en_titanet_large` with:

- embedding extraction
- zip-based speaker enrollment
- cosine-score verification
- cosine-score identification with `unknown` fallback below threshold
- Docker health checks that only pass once the model is loaded and the API is ready

## Endpoints

- `GET /healthz`
- `POST /v1/embeddings`
- `POST /v1/enroll`
- `POST /v1/verify`
- `POST /v1/identify`

All uploads are normalized to mono 16 kHz WAV with `ffmpeg` before inference, matching the model card input requirements.

## Quick start

1. Create `.env` from the example:

```bash
cp .env.example .env
```

2. Fill `HF_TOKEN` in `.env` if the model download requires it in your environment.

3. Start the stack:

```bash
docker compose up --build -d
```

4. Wait for readiness:

```bash
docker compose ps
curl http://localhost:8000/healthz
```

From inside this repo's devcontainer, use `http://host.docker.internal:8000` instead of `http://localhost:8000`, because the Docker daemon is running outside the devcontainer.

5. Enroll a speaker from a zip archive:

```bash
curl -sS \
  -F label=test_person \
  -F archive=@test_person.zip \
  http://localhost:8000/v1/enroll | jq
```

6. Identify a probe clip:

```bash
curl -sS \
  -F file=@test.opus \
  http://localhost:8000/v1/identify | jq
```

## Response shape

`/v1/embeddings` returns the model embedding as JSON:

```json
{
  "object": "speaker_embedding",
  "model": "nvidia/speakerverification_en_titanet_large",
  "device": "cuda",
  "dimension": 192,
  "duration_seconds": 2.0,
  "embedding": [0.0123, -0.0456]
}
```

`/v1/identify` returns sorted cosine scores and emits `unknown` when the best score is below threshold:

```json
{
  "object": "speaker_identification",
  "model": "nvidia/speakerverification_en_titanet_large",
  "threshold": 0.6,
  "predicted_label": "test_person",
  "unknown": false,
  "best_score": 0.87,
  "scores": [
    {
      "label": "test_person",
      "score": 0.87
    }
  ],
  "embedding_dimension": 192,
  "duration_seconds": 2.0
}
```

## Smoke test

Run the end-to-end smoke test from the devcontainer:

```bash
bash .devcontainer/smoke-test.sh
```
