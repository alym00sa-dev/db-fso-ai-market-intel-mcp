from server.main import health


def test_health_reports_status_and_version():
    result = health()
    assert result["status"] == "ok"
    assert result["version"]  # non-empty: git sha, env var, or package version fallback
    assert result["deployed_at"]


def test_resolve_version_env_var_takes_priority(monkeypatch):
    from server import main

    monkeypatch.setenv("GIT_COMMIT", "abc123def456")
    assert main._resolve_version() == "abc123def456"
