"""Exercise configuration and Windows entry points without downloading runtimes."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None


@pytest.fixture
def runtime_helper():
    spec = importlib.util.spec_from_file_location("windows_runtime", ROOT / "scripts/windows-runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings_file(tmp_path):
    settings = {
        "api_base_url": "https://example.org/v1",
        "api_key": "test-key-never-print",
        "model_name": "example-model",
        "install_directory": str(tmp_path / "运行目录"),
    }
    path = tmp_path / "安装配置.json"
    path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8-sig")
    return path


def test_apply_settings_updates_engine_and_web_without_losing_database(runtime_helper, settings_file, tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    engine_path = workspace / "config/engine_config.yaml"
    template = yaml.safe_load((ROOT / "yigraph/resources/engine_config.yaml").read_text(encoding="utf-8"))
    template["retrieval"]["database"]["neo4j"]["password"] = "keep-local-db-password"
    engine_path.write_text(yaml.safe_dump(template), encoding="utf-8")
    source = tmp_path / "source"
    (source / "web/frontend").mkdir(parents=True)
    models_path = source / "web/frontend/models.json"
    models_path.write_text(json.dumps([{"id": 8, "name": "existing", "api_key": "keep-existing"}]), encoding="utf-8")
    monkeypatch.setenv("YIGRAPH_HOME", str(workspace))

    runtime_helper.apply_config(settings_file, source)
    runtime_helper.apply_config(settings_file, source)

    engine = yaml.safe_load(engine_path.read_text(encoding="utf-8"))
    assert engine["reasoner"]["llm"]["openai"]["api_key"] == "test-key-never-print"
    assert engine["reasoner"]["llm"]["openai"]["base_url"] == "https://example.org/v1/"
    assert engine["retrieval"]["database"] == template["retrieval"]["database"]
    assert engine["retrieval"]["rag"] == template["retrieval"]["rag"]
    assert engine["retrieval"]["embedding"]["device"] == "cpu"
    models = json.loads(models_path.read_text(encoding="utf-8"))
    assert len(models) == 2
    assert models[0]["api_key"] == "keep-existing"
    assert models[1]["id"] == 9
    assert models[1]["name"] == "example-model"
    assert models[1]["api_key"] == "test-key-never-print"
    assert "test-key-never-print" not in capsys.readouterr().out


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
@pytest.mark.parametrize("action", ["Install", "Start", "Stop", "Configure", "Check"])
def test_plan_reads_unicode_config_without_writing_or_revealing_key(settings_file, tmp_path, action):
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/windows.ps1"), "-Config", str(settings_file), "-Action", action, "-Plan"],
        capture_output=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test-key-never-print" not in result.stdout + result.stderr
    assert not (tmp_path / "运行目录").exists()


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
def test_start_reports_missing_installation(settings_file):
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/windows.ps1"), "-Config", str(settings_file), "-Action", "Start"],
        capture_output=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 1
    assert "install.cmd" in result.stdout
    assert "test-key-never-print" not in result.stdout + result.stderr


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
def test_invalid_json_is_reported_without_powershell_traceback(tmp_path):
    settings = tmp_path / "broken.json"
    settings.write_text('{"api_key":', encoding="utf-8")
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/windows.ps1"), "-Config", str(settings), "-Plan"],
        capture_output=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 1
    assert "配置文件格式不正确" in result.stdout
    assert "CategoryInfo" not in result.stderr


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
def test_stop_does_not_terminate_reused_process_id(settings_file, tmp_path):
    runtime = tmp_path / "运行目录"
    runtime.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        (runtime / "processes.json").write_text(json.dumps({"web": process.pid, "web_started": 1, "neo4j": process.pid, "neo4j_started": 1}), encoding="utf-8")
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/windows.ps1"), "-Config", str(settings_file), "-Action", "Stop"],
            capture_output=True, encoding="utf-8", timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
def test_stop_terminates_only_recorded_child(settings_file, tmp_path):
    runtime = tmp_path / "运行目录"
    runtime.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        started = subprocess.check_output(
            [POWERSHELL, "-NoProfile", "-Command", f"(Get-Process -Id {process.pid}).StartTime.ToUniversalTime().Ticks"],
            text=True, timeout=20,
        ).strip()
        (runtime / "processes.json").write_text(json.dumps({"web": process.pid, "web_started": int(started), "neo4j": process.pid, "neo4j_started": int(started)}), encoding="utf-8")
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/windows.ps1"), "-Config", str(settings_file), "-Action", "Stop"],
            capture_output=True, encoding="utf-8", timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell is required")
def test_source_zip_without_git_is_accepted_without_download(tmp_path):
    source = tmp_path / "archive source"
    source.mkdir()
    shutil.copyfile(ROOT / "pyproject.toml", source / "pyproject.toml")
    runtime = tmp_path / "unused-runtime"
    bootstrap = str(ROOT / "scripts/windows-bootstrap.ps1").replace("'", "''")
    source_arg = str(source).replace("'", "''")
    runtime_arg = str(runtime).replace("'", "''")
    command = f". '{bootstrap}' -Root '{runtime_arg}' -Source '{source_arg}' -Profile Core -Plan; $env:PATH = $env:SystemRoot + '/System32'; Get-Checkout"
    result = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(str(source))
    assert not runtime.exists()
