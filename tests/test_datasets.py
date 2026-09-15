from server import datasets


def test_whitelist_has_16_datasets():
    assert len(datasets.DATASETS) == 16


def test_new_tables_whitelisted():
    for name in ("ornn_gpu_prices", "openrouter_rankings"):
        ds = datasets.get_dataset(name)
        assert ds.has_history is True


def test_model_benchmarks_overwritten_not_history():
    assert datasets.get_dataset("model_benchmarks").has_history is False


def test_girai_scores_tables_overwritten_single_edition_not_history():
    # load_girai.ipynb does a full .mode("overwrite") with a hardcoded single
    # EDITION per run (confirmed against the notebook) — nothing accumulates,
    # unlike girai_editions_comparison which is a fixed 2024-vs-2026 comparison
    # baked into its own source spreadsheet.
    for name in ("girai_scores", "girai_indicators", "girai_pillar_dimension_scores"):
        assert datasets.get_dataset(name).has_history is False


def test_hiring_tables_not_whitelisted():
    for name in ("hiring_jobs", "hiring_weekly", "hiring_weekly_breakdown", "hiring_snapshots"):
        assert name not in datasets.DATASETS


def test_unknown_dataset_fails_closed():
    try:
        datasets.get_dataset("not_a_real_table")
        assert False, "expected UnknownDatasetError"
    except datasets.UnknownDatasetError as exc:
        assert "not_a_real_table" in str(exc)


def test_full_name_uses_catalog_and_schema():
    ds = datasets.get_dataset("model_benchmarks")
    assert ds.full_name == "fso_market_intelligence.frontier_labs.model_benchmarks"
