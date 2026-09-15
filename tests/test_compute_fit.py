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


@pytest.fixture(autouse=True)
def _gpu_prices(stub_query_dataset):
    stub_query_dataset(cf, {"ornn_gpu_prices": GPU_PRICE_ROWS})


def test_requires_model_or_params_b():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit()


def test_rejects_unknown_precision():
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(params_b=7, precision="fp64")


def test_small_model_flagged_as_commodity_hardware():
    result = cf.compute_fit(params_b=7, precision="fp16")
    assert result["cheapest_fit"]["note"] == cf._CONSUMER_NOTE


def test_mid_model_needs_multi_gpu_at_fp16():
    result = cf.compute_fit(params_b=70, precision="fp16")
    assert "B200" in result["cheapest_fit"]["note"]
    assert result["cheapest_fit"]["n_gpus"] >= 2


def test_huge_model_needs_many_gpus():
    result = cf.compute_fit(params_b=405, precision="fp16")
    assert result["cheapest_fit"]["n_gpus"] > result_for_70b_n_gpus()


def result_for_70b_n_gpus():
    return cf.compute_fit(params_b=70, precision="fp16")["cheapest_fit"]["n_gpus"]


def test_int4_needs_far_less_vram_than_fp16():
    fp16 = cf.compute_fit(params_b=70, precision="fp16")
    int4 = cf.compute_fit(params_b=70, precision="int4")
    assert int4["total_gb"] < fp16["total_gb"]


def test_closed_weight_model_has_no_self_host_number(stub_query_dataset):
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "slug", "params_total_b", "context_window_tokens", "is_open_weights"],
                "rows": [
                    {
                        "name": "Closed Model",
                        "slug": "closed-model",
                        "params_total_b": 400,
                        "context_window_tokens": 128000,
                        "is_open_weights": False,
                    }
                ],
            },
        },
    )
    result = cf.compute_fit(model="Closed Model")
    assert result["self_hostable"] is False
    assert "closed-weight" in result["reason"].lower()


def test_unknown_model_fails_closed(stub_query_dataset):
    stub_query_dataset(
        cf,
        {
            "ornn_gpu_prices": GPU_PRICE_ROWS,
            "model_benchmarks": {"dataset": "model_benchmarks", "columns": [], "rows": []},
        },
    )
    with pytest.raises(cf.ComputeFitError):
        cf.compute_fit(model="Nonexistent Model 9000")
