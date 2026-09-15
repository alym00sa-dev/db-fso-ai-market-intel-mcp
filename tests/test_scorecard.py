import pytest

from server import scorecard

BENCHMARK_COLUMNS = [
    "name",
    "slug",
    "org",
    "params_total_b",
    "release_date",
    "is_open_weights",
    "intelligence_index",
    "gpqa_diamond",
    "hle",
    "price_blended_3to1",
    "cost_per_task_total",
]
ARENA_COLUMNS = ["model", "score"]  # load_arena_agent.ipynb renames HF's model_name -> model
OPENROUTER_COLUMNS = ["date", "model_permaslug", "total_tokens", "is_aggregate", "period", "rank", "captured_at"]
GPU_COLUMNS = ["date", "gpu_name", "index_value_usd_hr", "source", "source_updated_at", "captured_at"]

BENCHMARKS = [
    {
        "name": "Alpha 70B",
        "slug": "acme-alpha-70b",
        "org": "Acme",
        "params_total_b": 70,
        "release_date": "2026-06-01",
        "is_open_weights": True,
        "intelligence_index": 55,
        "gpqa_diamond": 0.6,
        "hle": 0.2,
        "price_blended_3to1": 2.0,
        "cost_per_task_total": 0.05,
    },
    {
        "name": "Beta Model X",
        "slug": "beta-x",
        "org": "Beta",
        "params_total_b": 400,
        "release_date": "2026-05-01",
        "is_open_weights": False,
        "intelligence_index": 80,
        "gpqa_diamond": 0.8,
        "hle": 0.4,
        "price_blended_3to1": 10.0,
        "cost_per_task_total": 0.5,
    },
    {
        "name": "Gamma Small 7B",
        "slug": "gamma-7b",
        "org": "Gamma",
        "params_total_b": 7,
        "release_date": "2026-04-01",
        "is_open_weights": True,
        "intelligence_index": 30,
        "gpqa_diamond": 0.3,
        "hle": 0.05,
        "price_blended_3to1": 0.5,
        "cost_per_task_total": 0.01,
    },
]
ARENA = [{"model": "Alpha 70B (High)", "score": 1300}, {"model": "Beta Model X (Adaptive)", "score": 1200}]
GPU_PRICES = [
    {"date": "2026-09-08", "gpu_name": "B200", "index_value_usd_hr": 6.5},
    {"date": "2026-09-08", "gpu_name": "RTX 5090", "index_value_usd_hr": 0.9},
]


@pytest.fixture
def wired_up(monkeypatch):
    """Wires scorecard.retrieval + compute_fit.retrieval to a small fake warehouse."""
    import server.compute_fit as cf
    import server.retrieval as retrieval

    def fake_describe_dataset(dataset):
        cols = {
            "model_benchmarks": BENCHMARK_COLUMNS,
            "arena_agent_leaderboard": ARENA_COLUMNS,
            "openrouter_rankings": OPENROUTER_COLUMNS,
            "ornn_gpu_prices": GPU_COLUMNS,
        }[dataset]
        return {"columns": [{"name": c, "type": "x", "comment": None} for c in cols]}

    def fake_query_dataset(dataset, columns=None, filters=None, limit=100, order_by=None, descending=False):
        if dataset == "model_benchmarks":
            rows = BENCHMARKS
            if filters:
                rows = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
            if columns:
                rows = [{c: r.get(c) for c in columns} for r in rows]
            return {"dataset": dataset, "columns": columns or [], "rows": rows[:limit]}
        if dataset == "arena_agent_leaderboard":
            return {"dataset": dataset, "columns": columns or [], "rows": ARENA[:limit]}
        if dataset == "openrouter_rankings":
            return {"dataset": dataset, "columns": columns or [], "rows": []}
        if dataset == "ornn_gpu_prices":
            return {"dataset": dataset, "columns": columns or [], "rows": GPU_PRICES}
        raise AssertionError(dataset)

    monkeypatch.setattr(retrieval, "describe_dataset", fake_describe_dataset)
    monkeypatch.setattr(retrieval, "query_dataset", fake_query_dataset)


def test_open_weight_model_gets_full_card(wired_up):
    card = scorecard.model_scorecard("Alpha 70B")
    assert card["identity"]["name"] == "Alpha 70B"
    assert card["identity"]["organization"] == "Acme"  # sourced from the real "org" column
    assert card["measured"]["intelligence_index"] == 55.0
    assert card["measured"]["peer_set_size"] == 3
    assert card["stated"]["available"] is True
    assert card["stated"]["arena_model_name"] == "Alpha 70B (High)"
    assert card["revealed"]["available"] is False
    assert card["revealed"]["reason"] == "not tracked on OpenRouter"


def test_stated_percentile_higher_than_measured_flags_hyped(wired_up):
    card = scorecard.model_scorecard("Alpha 70B")
    # Alpha's arena score is top of the (2-model) arena set -> percentile 1.0,
    # well above its measured percentile among the 3-model benchmark peer set.
    assert "hyped" in card["flags"]


def test_closed_weight_model_has_no_self_host_number(wired_up):
    card = scorecard.model_scorecard("Beta Model X")
    assert card["identity"]["is_open_weights"] is False
    assert card["cost"]["self_host"]["self_hostable"] is False


def test_open_weight_model_gets_compute_fit_delegated_self_host(wired_up):
    card = scorecard.model_scorecard("Gamma Small 7B")
    self_host = card["cost"]["self_host"]
    assert self_host["self_hostable"] is True
    assert "cheapest_fit" in self_host  # came from compute_fit, not inlined math


def test_model_not_found_fails_closed(wired_up):
    with pytest.raises(scorecard.ModelScorecardError):
        scorecard.model_scorecard("Totally Not A Model")


def test_model_matched_by_normalized_key_ignores_case_and_whitespace(wired_up):
    card = scorecard.model_scorecard("  alpha   70b  ")
    assert card["identity"]["name"] == "Alpha 70B"
