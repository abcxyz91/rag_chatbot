import importlib
import sys
import tomllib
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_main_imports_without_initializing_rag_backends(monkeypatch, tmp_path):
    """Import the application while replacing side-effectful RAG tool constructors."""

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai"))

    class StubTool:
        def __init__(self, *args, **kwargs):
            pass

    crewai_tools = ModuleType("crewai_tools")
    for tool_name in (
        "DirectoryReadTool",
        "PDFSearchTool",
        "DOCXSearchTool",
        "CSVSearchTool",
    ):
        setattr(crewai_tools, tool_name, StubTool)

    monkeypatch.setitem(sys.modules, "crewai_tools", crewai_tools)

    main = importlib.import_module("rag_chatbot.main")

    assert callable(main.HRCrew)
    assert callable(main.kickoff)


def test_general_task_uses_runtime_input_name():
    task_config = (
        REPO_ROOT
        / "src"
        / "rag_chatbot"
        / "crews"
        / "general_crew"
        / "config"
        / "tasks.yaml"
    ).read_text(encoding="utf-8")

    assert "{user_query}" in task_config
    assert "{question}" not in task_config


def test_runtime_dependency_and_documented_cli_are_aligned():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    scripts = pyproject["project"]["scripts"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert any(dependency.startswith("python-dotenv") for dependency in dependencies)
    assert "ollama>=0.4.0,<1.0.0" in dependencies
    assert "crewai[tools]~=0.157.0" in dependencies
    assert "chromadb==0.5.23" in dependencies
    assert "embedchain==0.1.128" in dependencies
    assert "kickoff" in scripts
    assert "rag_chatbot-kickoff" not in readme


def test_domain_crews_use_embedchain_chroma_config():
    crews_root = REPO_ROOT / "src" / "rag_chatbot" / "crews"

    for domain in ("accounting", "hr", "legal"):
        crew_source = (crews_root / f"{domain}_crew" / f"{domain}_crew.py").read_text(
            encoding="utf-8"
        )

        assert f"domain_rag_config('{domain}')" in crew_source
        assert "./knowledge_base/" not in crew_source
        assert "./chroma_db/" not in crew_source
