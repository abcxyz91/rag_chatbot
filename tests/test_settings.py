from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_chatbot.settings import Settings, domain_rag_config, get_settings


def test_defaults_are_absolute_and_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)

    assert settings.knowledge_base_path.is_absolute()
    assert settings.chroma_path.is_absolute()
    assert settings.domain_knowledge_path("accounting").is_dir()


def test_relative_environment_paths_resolve_from_project_root(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "custom/knowledge")
    monkeypatch.setenv("CHROMA_PATH", "custom/chroma")

    settings = Settings(_env_file=None)

    assert settings.knowledge_base_path.is_absolute()
    assert settings.knowledge_base_path.parts[-2:] == ("custom", "knowledge")
    assert settings.chroma_path.parts[-2:] == ("custom", "chroma")


def test_environment_overrides_models_and_runtime_options(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434/")
    monkeypatch.setenv("ROUTER_MODEL", "router:test")
    monkeypatch.setenv("ANSWER_MODEL", "answer:test")
    monkeypatch.setenv("EMBED_MODEL", "embed:test")
    monkeypatch.setenv("DEFAULT_TOP_K", "8")
    monkeypatch.setenv("STREAMING_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.ollama_url == "http://ollama.internal:11434"
    assert settings.ollama_model(settings.router_model) == "ollama/router:test"
    assert settings.answer_model == "answer:test"
    assert settings.embed_model == "embed:test"
    assert settings.default_top_k == 8
    assert settings.streaming_enabled is False


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValidationError):
        Settings(default_top_k=0, _env_file=None)


def test_startup_validation_reports_missing_domain_folders(tmp_path):
    settings = Settings(
        knowledge_base_path=tmp_path / "missing",
        chroma_path=tmp_path / "chroma",
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match="Missing knowledge-base directories"):
        settings.validate_startup()


def test_startup_validation_creates_chroma_storage(tmp_path):
    knowledge = tmp_path / "knowledge"
    for domain in ("accounting", "hr", "legal"):
        (knowledge / domain).mkdir(parents=True)

    settings = Settings(
        knowledge_base_path=knowledge,
        chroma_path=tmp_path / "indexes",
        _env_file=None,
    )
    settings.validate_startup()

    assert settings.chroma_path.is_dir()


def test_domain_rag_config_uses_typed_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("EMBED_MODEL", "embed:test")
    get_settings.cache_clear()

    config = domain_rag_config("legal")

    assert config["embedding_model"]["config"]["model"] == "embed:test"
    assert Path(config["vectordb"]["config"]["dir"]) == tmp_path / "chroma" / "legal"
    get_settings.cache_clear()
