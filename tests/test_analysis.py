from server import analysis


def test_capability_cost_pareto_separates_frontier_from_dominated(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "price_blended_3to1"],
                "rows": [
                    {"name": "Cheap-Low", "intelligence_index": 30, "price_blended_3to1": 0.01},
                    {"name": "Mid", "intelligence_index": 50, "price_blended_3to1": 0.02},
                    {"name": "Dominated", "intelligence_index": 40, "price_blended_3to1": 0.03},
                    {"name": "Best", "intelligence_index": 90, "price_blended_3to1": 0.1},
                ],
            }
        },
    )
    result = analysis.capability_cost_pareto()
    assert result["cost_column"] == "price_blended_3to1"
    assert [r["model"] for r in result["frontier"]] == ["Cheap-Low", "Mid", "Best"]
    assert [r["model"] for r in result["dominated"]] == ["Dominated"]


def test_capability_cost_pareto_tie_on_cost_prefers_higher_score(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "price_blended_3to1"],
                "rows": [
                    {"name": "SameCostLowScore", "intelligence_index": 20, "price_blended_3to1": 0.05},
                    {"name": "SameCostHighScore", "intelligence_index": 60, "price_blended_3to1": 0.05},
                ],
            }
        },
    )
    result = analysis.capability_cost_pareto()
    assert [r["model"] for r in result["frontier"]] == ["SameCostHighScore"]
    assert [r["model"] for r in result["dominated"]] == ["SameCostLowScore"]


def test_capability_cost_pareto_skips_missing_or_non_positive_cost(stub_query_dataset):
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "intelligence_index", "price_blended_3to1"],
                "rows": [
                    {"name": "NoCost", "intelligence_index": 50, "price_blended_3to1": None},
                    {"name": "ZeroCost", "intelligence_index": 50, "price_blended_3to1": 0},
                    {"name": "NegCost", "intelligence_index": 50, "price_blended_3to1": -1},
                    {"name": "Good", "intelligence_index": 50, "price_blended_3to1": 1.0},
                ],
            }
        },
    )
    result = analysis.capability_cost_pareto()
    assert [r["model"] for r in result["frontier"]] == ["Good"]


def test_capability_cost_pareto_score_column_is_caller_specified(stub_query_dataset):
    """score_column isn't hardcoded to intelligence_index — any live column works."""
    stub_query_dataset(
        analysis,
        {
            "model_benchmarks": {
                "dataset": "model_benchmarks",
                "columns": ["name", "coding_index", "price_blended_3to1"],
                "rows": [
                    {"name": "A", "coding_index": 40, "price_blended_3to1": 0.02},
                    {"name": "B", "coding_index": 70, "price_blended_3to1": 0.05},
                ],
            }
        },
    )
    result = analysis.capability_cost_pareto(score_column="coding_index")
    assert result["score_column"] == "coding_index"
    assert [r["model"] for r in result["frontier"]] == ["A", "B"]
