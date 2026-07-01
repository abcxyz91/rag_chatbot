"""Typed, environment-driven application settings and path helpers."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent


def _project_root() -> Path:
    for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.home() / ".rag_chatbot"


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    knowledge_base_path: Path = Field(default=PACKAGE_ROOT / "knowledge_base")
    chroma_path: Path = Field(default=PROJECT_ROOT / "chroma_db")
    ollama_base_url: AnyHttpUrl = Field(default="http://localhost:11434")
    router_model: str = Field(default="gemma4:e2b-it-bf16", min_length=1)
    answer_model: str = Field(default="gemma4:e4b-it-bf16", min_length=1)
    embed_model: str = Field(default="embeddinggemma:300m", min_length=1)
    default_top_k: int = Field(default=5, ge=1)
    streaming_enabled: bool = True

    def model_post_init(self, __context: object) -> None:
        self.knowledge_base_path = self._absolute(self.knowledge_base_path)
        self.chroma_path = self._absolute(self.chroma_path)

    @staticmethod
    def _absolute(path: Path) -> Path:
        path = path.expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def ollama_url(self) -> str:
        return str(self.ollama_base_url).rstrip("/")

    def ollama_model(self, model: str) -> str:
        return model if "/" in model else f"ollama/{model}"

    def domain_knowledge_path(self, domain: Literal["accounting", "hr", "legal"]) -> Path:
        return self.knowledge_base_path / domain

    def domain_chroma_path(self, domain: Literal["accounting", "hr", "legal"]) -> Path:
        return self.chroma_path / domain

    def validate_startup(self, *, create_chroma: bool = True) -> None:
        """Validate required source folders and prepare persistent storage."""

        missing = [
            str(self.domain_knowledge_path(domain))
            for domain in ("accounting", "hr", "legal")
            if not self.domain_knowledge_path(domain).is_dir()
        ]
        if missing:
            raise RuntimeError(
                "Missing knowledge-base directories: " + ", ".join(missing)
            )

        if create_chroma:
            try:
                self.chroma_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot create CHROMA_PATH at {self.chroma_path}: {exc}"
                ) from exc


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings instance per process."""

    return Settings()


def domain_rag_config(domain: Literal["accounting", "hr", "legal"]) -> dict:
    """Build Embedchain configuration for a domain's persistent index."""

    settings = get_settings()
    return {
        "embedding_model": {
            "provider": "ollama",
            "config": {
                "model": settings.embed_model,
                "base_url": settings.ollama_url,
            },
        },
        "vectordb": {
            "provider": "chroma",
            "config": {
                "dir": str(settings.domain_chroma_path(domain)),
                "allow_reset": True,
            },
        },
    }
