"""Covers the on-behalf-of-user auth fix: _client()/_actor() must never cache,
and must build from the forwarded token when deployed vs. local CLI auth
otherwise. All WorkspaceClient/HTTP-header access is mocked — this is a unit
test of db.py's own logic, not an integration test against a real workspace."""

from __future__ import annotations

from unittest import mock

import pytest

from server import db


@pytest.fixture(autouse=True)
def _clean_app_name_env(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)


def test_client_falls_back_to_plain_workspaceclient_locally():
    with mock.patch.object(db, "WorkspaceClient") as MockWC:
        db._client()
        MockWC.assert_called_once_with()


def test_client_requires_forwarded_token_when_deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "my-app")
    with mock.patch.object(db, "get_http_headers", return_value={}):
        with pytest.raises(db.QueryError, match="x-forwarded-access-token"):
            db._client()


def test_client_uses_forwarded_token_when_deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "my-app")
    with mock.patch.object(
        db, "get_http_headers", return_value={"x-forwarded-access-token": "tok-123"}
    ):
        with mock.patch.object(db, "WorkspaceClient") as MockWC:
            db._client()
            MockWC.assert_called_once_with(token="tok-123", auth_type="pat")


def test_client_is_not_cached_across_calls_with_different_users(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "my-app")
    headers = [{"x-forwarded-access-token": "user-a"}, {"x-forwarded-access-token": "user-b"}]
    with mock.patch.object(db, "get_http_headers", side_effect=headers):
        with mock.patch.object(db, "WorkspaceClient") as MockWC:
            db._client()
            db._client()
            assert MockWC.call_args_list == [
                mock.call(token="user-a", auth_type="pat"),
                mock.call(token="user-b", auth_type="pat"),
            ]


def test_actor_resolves_current_user():
    with mock.patch.object(db, "WorkspaceClient") as MockWC:
        MockWC.return_value.current_user.me.return_value.user_name = "alice@example.com"
        assert db._actor() == "alice@example.com"


def test_actor_falls_back_to_unknown_without_raising():
    with mock.patch.object(db, "WorkspaceClient") as MockWC:
        MockWC.return_value.current_user.me.side_effect = Exception("boom")
        assert db._actor() == "unknown"


def test_warehouse_id_missing_raises_query_error(monkeypatch):
    monkeypatch.delenv(db._WAREHOUSE_ID_ENV_VAR, raising=False)
    with pytest.raises(db.QueryError, match=db._WAREHOUSE_ID_ENV_VAR):
        db._warehouse_id()


def test_run_query_end_to_end_with_mocked_client(monkeypatch):
    monkeypatch.setenv(db._WAREHOUSE_ID_ENV_VAR, "wh-123")

    class FakeStatus:
        state = db.StatementState.SUCCEEDED
        error = None

    class FakeColumn:
        def __init__(self, name):
            self.name = name

    class FakeResponse:
        statement_id = "stmt-1"
        status = FakeStatus()

        class manifest:
            class schema:
                columns = [FakeColumn("a"), FakeColumn("b")]

        class result:
            data_array = [[1, 2]]

    with mock.patch.object(db, "WorkspaceClient") as MockWC:
        MockWC.return_value.statement_execution.execute_statement.return_value = FakeResponse()
        result = db.run_query("SELECT :x", {"x": 1})

    assert result.columns == ["a", "b"]
    assert result.rows == [[1, 2]]
    assert result.as_dicts() == [{"a": 1, "b": 2}]
