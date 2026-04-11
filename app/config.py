from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_port: int = 8000
    model_id: str = "nvidia/speakerverification_en_titanet_large"
    model_device: str = "cuda"
    model_cache_dir: str = "/cache/huggingface"
    hf_home: str = "/cache/huggingface"
    hf_hub_disable_telemetry: int = 1
    hf_token: str = ""
    identification_threshold: float = 0.60
    verification_threshold: float = 0.60
    store_path: str = "/data/speaker_store.json"
    max_upload_bytes: int = 104857600
    max_enroll_files: int = 64
    log_level: str = "info"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
