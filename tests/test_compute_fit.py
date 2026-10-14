"""Tests follow compute_fit_changes.md's T1-T20 table (Llama 3.3 70B reference case)."""

import pytest

from server import compute_fit as cf

GPU_PRICE_ROWS = {
    "dataset": "ornn_gpu_prices",
    "columns": ["date", "gpu_name", "index_value_usd_hr"],
    "rows": [
        {"date": "2026-09-01", "gpu_name": "B200", "index_value_usd_hr": 6.5},
        {"date": "2026-09-01", "gpu_name": "H200", "index_value_usd_hr": 3.8},
        {"date": "2026-09-01", "gpu_name": "H100 SXM", "index_value_usd_hr": 2.5},
        {"date": "2026-09-01", "gpu_name": "A100 SXM4", "index_value_usd_hr": 1.8},
        {"date": "2026-09-01", "gpu_name": "RTX 5090", "index_value_usd_hr": 0.9},
    ],
}

# The spec's reference case: Llama 3.3 70B, passed via explicit architecture
# args (the "user_supplied" kv_source tier) rather than a model_benchmarks
# lookup, so most tests don't need to mock model_benchmarks at all.
LLAMA = dict(
    params_b=70.6,
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    num_attention_heads=64,
    context_length=8192,
    overhead_gb_per_gpu=3.0,
    usable_fraction=0.9,
)


def _config(result, gpu_name, n):
    return next(
        c
        for c in result["configurations"]
        if c["gpu_name"] == gpu_name and c["n_gpus_per_instance"] == n
    )


@pytest.fixture(autouse=True)
def _gpu_prices(stub_query_dataset):
    stub_query_dataset(cf, {"ornn_gpu_prices": GPU_PRICE_ROWS})


# --- basic input validation (not in the T-table, but required by change 10) ---


def test_requires_model_or_params_b():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit()


def test_rejects_unknown_precision():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, precision="fp64")


def test_rejects_unknown_kv_precision():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, kv_precision="fp64")


def test_rejects_non_positive_params_b():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=0)


def test_rejects_batch_size_below_one():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, batch_size=0)


def test_rejects_context_length_below_one():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, context_length=0)


def test_rejects_usable_fraction_out_of_range():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, usable_fraction=1.5)
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, usable_fraction=0)


def test_small_model_flagged_as_commodity_hardware():
    result = cf.compute_fit(params_b=7, precision="fp16")
    assert result["cheapest_fit"]["note"] == cf._CONSUMER_NOTE


# --- T1: KV bytes/token at bf16 ---


def test_t1_kv_bytes_per_token_bf16():
    result = cf.compute_fit(**LLAMA, precision="bf16")
    assert result["assumptions"]["kv_bytes_per_token"] == pytest.approx(327680, abs=1)
    assert result["kv_gb"] == pytest.approx(2.684, abs=0.05)


# --- T2: KV per request at 131,072 context ---


def test_t2_kv_per_request_at_full_context():
    kwargs = {**LLAMA, "context_length": 131072}
    result = cf.compute_fit(**kwargs, precision="bf16")
    assert result["kv_gb"] == pytest.approx(42.95, abs=0.05)


# --- T3: weights at bf16 / fp8 / int4 ---


def test_t3_weights_by_precision():
    bf16 = cf.compute_fit(**LLAMA, precision="bf16")
    fp8 = cf.compute_fit(**LLAMA, precision="fp8")
    int4 = cf.compute_fit(**LLAMA, precision="int4")
    assert bf16["weights_gb"] == pytest.approx(141.2, abs=0.05)
    assert fp8["weights_gb"] == pytest.approx(70.6, abs=0.05)
    assert int4["weights_gb"] == pytest.approx(39.71, abs=0.05)


# --- T4: 1x B200, bf16 weights ---


def test_t4_1x_b200_bf16():
    result = cf.compute_fit(**LLAMA, precision="bf16")
    cfg = _config(result, "B200", 1)
    assert cfg["kv_pool_gb"] == pytest.approx(28.6, abs=0.05)
    assert cfg["max_concurrent_requests"] == 10


# --- T5: 2x H100, bf16 weights -> does not load ---


def test_t5_2x_h100_bf16_does_not_load():
    result = cf.compute_fit(**LLAMA, precision="bf16")
    cfg = _config(result, "H100 SXM", 2)
    assert cfg["kv_pool_gb"] < 0
    assert cfg["status"] == "does_not_load"


# --- T6: 2x H100, fp8 weights ---


def test_t6_2x_h100_fp8():
    result = cf.compute_fit(**LLAMA, precision="fp8")
    cfg = _config(result, "H100 SXM", 2)
    assert cfg["kv_pool_gb"] == pytest.approx(67.4, abs=0.05)
    assert cfg["max_concurrent_requests"] == 25
    assert cfg["weights_share_of_usable"] == pytest.approx(0.49, abs=0.01)


# --- T7: 2x H100, fp8 weights, kv_precision="fp8" ---


def test_t7_2x_h100_fp8_weights_fp8_kv():
    result = cf.compute_fit(**LLAMA, precision="fp8", kv_precision="fp8")
    cfg = _config(result, "H100 SXM", 2)
    assert cfg["max_concurrent_requests"] == 50


# --- T8: 1x H200, fp8 weights ---


def test_t8_1x_h200_fp8():
    result = cf.compute_fit(**LLAMA, precision="fp8")
    cfg = _config(result, "H200", 1)
    assert cfg["kv_pool_gb"] == pytest.approx(53.3, abs=0.05)
    assert cfg["max_concurrent_requests"] == 19


# --- T9: 1x H100, int4 weights ---


def test_t9_1x_h100_int4():
    result = cf.compute_fit(**LLAMA, precision="int4")
    cfg = _config(result, "H100 SXM", 1)
    assert cfg["kv_pool_gb"] == pytest.approx(29.29, abs=0.05)
    assert cfg["max_concurrent_requests"] == 10


# --- T10: 4x H100, bf16 weights ---


def test_t10_4x_h100_bf16():
    result = cf.compute_fit(**LLAMA, precision="bf16")
    cfg = _config(result, "H100 SXM", 4)
    assert cfg["kv_pool_gb"] == pytest.approx(134.8, abs=0.05)
    assert cfg["max_concurrent_requests"] == 50


# --- T11: 2x H100 fp8, batch_size=30 ---


def test_t11_replicas_and_total_gpus():
    kwargs = {**LLAMA, "batch_size": 30}
    result = cf.compute_fit(**kwargs, precision="fp8")
    cfg = _config(result, "H100 SXM", 2)
    assert cfg["replicas"] == 2
    assert cfg["total_gpus"] == 4


# --- T12: num_attention_heads=28 -> n=8 skipped ---


def test_t12_invalid_shard_sizes_skipped():
    kwargs = {**LLAMA, "num_attention_heads": 28}
    result = cf.compute_fit(**kwargs, precision="bf16")
    n_values = {c["n_gpus_per_instance"] for c in result["configurations"]}
    assert 8 not in n_values
    assert {1, 2, 4}.issubset(n_values)


# --- T13: precision="bf16" accepted ---


def test_t13_bf16_accepted():
    result = cf.compute_fit(params_b=7, precision="bf16")
    assert result["precision"] == "bf16"


# --- T14: forgiving model name lookup ---


def _model_benchmarks_route(rows):
    def route(dataset, columns=None, filters=None, limit=100, order_by=None, descending=False):
        result_rows = rows
        if filters:
            result_rows = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
        if columns:
            result_rows = [{c: r.get(c) for c in columns} for r in result_rows]
        return {"dataset": dataset, "columns": columns or [], "rows": result_rows[:limit]}

    return route


LLAMA_BENCHMARK_COLUMNS = ["name", "slug", "params_total_b", "context_window_tokens", "is_open_weights"]
LLAMA_ROW = {
    "name": "Llama 3.3 70B Instruct",
    "slug": "llama-3.3-70b-instruct",
    "params_total_b": 70.6,
    "context_window_tokens": 131072,
    "is_open_weights": True,
}


def test_t14_forgiving_name_match(stub_describe_dataset, stub_query_dataset):
    stub_describe_dataset(cf, {"model_benchmarks": LLAMA_BENCHMARK_COLUMNS})
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": _model_benchmarks_route([LLAMA_ROW]),
        },
    )
    result = cf.compute_fit(model="llama 3.3 70b")
    assert result["model"] == "Llama 3.3 70B Instruct"


# --- T15: ambiguous name ---


def test_t15_ambiguous_name_raises(stub_describe_dataset, stub_query_dataset):
    rows = [
        LLAMA_ROW,
        {**LLAMA_ROW, "name": "Llama 3.3 70B Instruct Turbo", "slug": "llama-3.3-70b-instruct-turbo"},
    ]
    stub_describe_dataset(cf, {"model_benchmarks": LLAMA_BENCHMARK_COLUMNS})
    stub_query_dataset(
        cf,
        {"ornn_gpu_prices": GPU_PRICE_ROWS, "model_benchmarks": _model_benchmarks_route(rows)},
    )
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(model="llama 3.3 70b")


# --- T16: model= lookup without context_length defaults to 8192 ---


def test_t16_model_lookup_defaults_context_to_8192(stub_describe_dataset, stub_query_dataset):
    stub_describe_dataset(cf, {"model_benchmarks": LLAMA_BENCHMARK_COLUMNS})
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": _model_benchmarks_route([LLAMA_ROW]),
        },
    )
    result = cf.compute_fit(model="Llama 3.3 70B Instruct")
    assert result["assumptions"]["context_length"] == 8192
    assert "max_context_stress" in result
    assert result["max_context_stress"]["full_context_window_tokens"] == 131072


# --- T17: unsorted price rows -> latest still selected correctly ---


def test_t17_unsorted_price_rows_latest_selected(monkeypatch):
    rows = [
        {"date": "2026-01-01", "gpu_name": "B200", "index_value_usd_hr": 1.0},
        {"date": "2026-03-01", "gpu_name": "B200", "index_value_usd_hr": 9.0},
        {"date": "2026-02-01", "gpu_name": "B200", "index_value_usd_hr": 5.0},
    ]

    def route(dataset, columns=None, filters=None, limit=100, order_by=None, descending=False):
        result_rows = rows
        if order_by == "date" and descending:
            result_rows = sorted(rows, key=lambda r: r["date"], reverse=True)
        return {"dataset": dataset, "columns": columns or [], "rows": result_rows[:limit]}

    monkeypatch.setattr(cf.retrieval, "query_dataset", route)
    result = cf.compute_fit(params_b=7, precision="bf16")
    cfg = _config(result, "B200", 1)
    assert cfg["price_usd_hr_per_gpu"] == 9.0


# --- T18: architecture missing -> estimated, with range ---


def test_t18_estimated_kv_source_and_range():
    result = cf.compute_fit(params_b=70.6, precision="bf16")
    assert result["assumptions"]["kv_source"] == "estimated"
    assert "kv_mb_per_token_range" in result
    assert result["kv_mb_per_token_range"][0] == pytest.approx(0.326, abs=0.01)


# --- T19: non-default overhead_factor is ignored but flagged ---


def test_t19_deprecated_overhead_factor_ignored_but_flagged():
    base = cf.compute_fit(**LLAMA, precision="bf16")
    with_deprecated = cf.compute_fit(**LLAMA, precision="bf16", overhead_factor=1.5)
    assert with_deprecated["total_gb"] == base["total_gb"]
    assert any("deprecated" in c.lower() for c in with_deprecated["caveats"])


# --- T20: closed-weight model unchanged ---


def test_t20_closed_weight_model_unchanged(stub_describe_dataset, stub_query_dataset):
    closed_row = {
        "name": "Closed Model",
        "slug": "closed-model",
        "params_total_b": 400,
        "context_window_tokens": 128000,
        "is_open_weights": False,
    }
    stub_describe_dataset(cf, {"model_benchmarks": LLAMA_BENCHMARK_COLUMNS})
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": _model_benchmarks_route([closed_row]),
        },
    )
    result = cf.compute_fit(model="Closed Model")
    assert result["self_hostable"] is False
    assert "closed-weight" in result["reason"].lower()


def test_unknown_model_fails_closed(stub_describe_dataset, stub_query_dataset):
    stub_describe_dataset(cf, {"model_benchmarks": LLAMA_BENCHMARK_COLUMNS})
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": _model_benchmarks_route([]),
        },
    )
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(model="Nonexistent Model 9000")
