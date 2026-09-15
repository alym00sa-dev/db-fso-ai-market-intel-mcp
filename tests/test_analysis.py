from server import analysis


def test_model_roi_ranks_by_capability_per_dollar(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "price_blended_3to1"],
                "rows": [
                    {"name": "A", "intelligence_index": 50, "price_blended_3to1": 1.0},
                    {"name": "B", "intelligence_index": 80, "price_blended_3to1": 2.0},
                ],
            }
        },
    )
    result = analysis.model_roi()
    assert [r["model"] for r in result["ranking"]] == ["A", "B"]
    assert result["ranking"][0]["capability_per_dollar"] == 50.0


def test_model_roi_skips_rows_with_missing_or_non_positive_price(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "price_blended_3to1"],
                "rows": [
                    {"name": "NoPrice", "intelligence_index": 50, "price_blended_3to1": None},
                    {"name": "ZeroPrice", "intelligence_index": 50, "price_blended_3to1": 0},
                    {"name": "NegPrice", "intelligence_index": 50, "price_blended_3to1": -1},
                    {"name": "Good", "intelligence_index": 50, "price_blended_3to1": 1.0},
                ],
            }
        },
    )
    result = analysis.model_roi()
    assert [r["model"] for r in result["ranking"]] == ["Good"]


def test_cost_efficiency_frontier_separates_frontier_from_dominated(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "cost_per_task_total"],
                "rows": [
                    {"name": "Cheap-Low", "intelligence_index": 30, "cost_per_task_total": 0.01},
                    {"name": "Mid", "intelligence_index": 50, "cost_per_task_total": 0.02},
                    {"name": "Dominated", "intelligence_index": 40, "cost_per_task_total": 0.03},
                    {"name": "Best", "intelligence_index": 90, "cost_per_task_total": 0.1},
                ],
            }
        },
    )
    result = analysis.cost_efficiency_frontier()
    assert [r["model"] for r in result["frontier"]] == ["Cheap-Low", "Mid", "Best"]
    assert [r["model"] for r in result["dominated"]] == ["Dominated"]


def test_cost_efficiency_frontier_tie_on_cost_prefers_higher_score(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "cost_per_task_total"],
                "rows": [
                    {"name": "SameCostLowScore", "intelligence_index": 20, "cost_per_task_total": 0.05},
                    {"name": "SameCostHighScore", "intelligence_index": 60, "cost_per_task_total": 0.05},
                ],
            }
        },
    )
    result = analysis.cost_efficiency_frontier()
    assert [r["model"] for r in result["frontier"]] == ["SameCostHighScore"]
    assert [r["model"] for r in result["dominated"]] == ["SameCostLowScore"]
