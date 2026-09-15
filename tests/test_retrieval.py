"""retrieval.py is the only place SQL gets built against the whitelist, so its
column validation / filter binding / order_by / limit-clamping logic is worth
testing directly against the real code path (not through a query_dataset stub)."""

from __future__ import annotations

from unittest import mock

import pytest

from server import db, retrieval
from server.datasets import UnknownDatasetError


def test_list_datasets_returns_all_19():
    assert len(retrieval.list_datasets()) == 19


def test_describe_dataset_unknown_raises(patch_get_columns):
    patch_get_columns()
    with pytest.raises(UnknownDatasetError):
        retrieval.describe_dataset("nonexistent")


def test_describe_dataset_returns_live_columns(patch_get_columns):
    columns = patch_get_columns()
    described = retrieval.describe_dataset("ornn_gpu_prices")
    names = {c["name"] for c in described["columns"]}
    assert names == set(columns["ornn_gpu_prices"])
    assert "gpu_name" in names and "index_value_usd_hr" in names


def test_query_dataset_rejects_unknown_column(patch_get_columns):
    patch_get_columns()
    with pytest.raises(retrieval.UnknownColumnError):
        retrieval.query_dataset("model_benchmarks", columns=["not_a_real_column"])


def test_query_dataset_rejects_unknown_filter_column(patch_get_columns):
    patch_get_columns()
    with pytest.raises(retrieval.UnknownColumnError):
        retrieval.query_dataset("model_benchmarks", filters={"not_a_real_column": 1})


def test_query_dataset_rejects_unknown_order_by(patch_get_columns):
    patch_get_columns()
    with pytest.raises(retrieval.UnknownColumnError):
        retrieval.query_dataset("ornn_gpu_prices", order_by="not_a_real_column")


def test_query_dataset_builds_parameterized_equality_and_in_filters(patch_get_columns):
    patch_get_columns()

    captured = {}

    def spy(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return db.QueryResult(columns=["name"], rows=[["Alpha"]])

    with mock.patch.object(db, "run_query", side_effect=spy):
        retrieval.query_dataset(
            "model_benchmarks",
            columns=["name"],
            filters={"is_open_weights": True, "organization": ["Acme", "Beta"]},
        )

    sql = captured["sql"]
    assert "`is_open_weights` = :f0" in sql
    assert "`organization` IN (:f1_0, :f1_1)" in sql
    assert captured["params"] == {"f0": True, "f1_0": "Acme", "f1_1": "Beta"}
    # values are bound as parameters, never interpolated into the SQL string
    assert "Acme" not in sql and "Beta" not in sql


def test_query_dataset_order_by_and_limit_clamping(patch_get_columns):
    patch_get_columns()

    captured = {}

    def spy(sql, params=None):
        captured["sql"] = sql
        return db.QueryResult(columns=["date"], rows=[])

    with mock.patch.object(db, "run_query", side_effect=spy):
        retrieval.query_dataset(
            "ornn_gpu_prices", order_by="date", descending=True, limit=999999
        )

    assert "ORDER BY `date` DESC" in captured["sql"]
    assert f"LIMIT {retrieval.MAX_LIMIT}" in captured["sql"]  # clamped


def test_query_dataset_limit_floor_is_one(patch_get_columns):
    patch_get_columns()
    captured = {}

    def spy(sql, params=None):
        captured["sql"] = sql
        return db.QueryResult(columns=[], rows=[])

    with mock.patch.object(db, "run_query", side_effect=spy):
        retrieval.query_dataset("ornn_gpu_prices", limit=0)

    assert "LIMIT 1" in captured["sql"]


def test_query_dataset_shapes_rows_as_dicts(patch_get_columns, fake_db_run_query):
    patch_get_columns()
    fake_db_run_query(
        columns=["gpu_name", "index_value_usd_hr"],
        rows=[["B200", 6.5], ["RTX 5090", 0.9]],
    )
    result = retrieval.query_dataset("ornn_gpu_prices")
    assert result["rows"] == [
        {"gpu_name": "B200", "index_value_usd_hr": 6.5},
        {"gpu_name": "RTX 5090", "index_value_usd_hr": 0.9},
    ]
