from server import equity

CLASSIFICATIONS = {
    "dataset": "girai_country_classifications",
    "columns": ["iso3", "country", "region"],
    "rows": [
        {"iso3": "KEN", "country": "Kenya", "region": "Africa"},
        {"iso3": "NGA", "country": "Nigeria", "region": "Africa"},
        {"iso3": "USA", "country": "United States", "region": "Americas"},
        {"iso3": "PSE", "country": "Palestine", "region": "Asia"},  # only in classifications
    ],
}
GIRAI = {
    "dataset": "girai_scores",
    "columns": ["iso3", "girai"],
    "rows": [
        {"iso3": "KEN", "girai": 40.0},
        {"iso3": "NGA", "girai": 30.0},
        {"iso3": "USA", "girai": 80.0},
    ],
}
OXFORD = {
    "dataset": "oxford_ai_readiness",
    "columns": ["iso3", "total_score"],
    "rows": [
        {"iso3": "KEN", "total_score": 50.0},
        {"iso3": "USA", "total_score": 90.0},
        # Nigeria and Palestine missing from Oxford entirely — the two country
        # lists genuinely don't match (135 GIRAI vs ~195 Oxford).
    ],
}


def _stub(stub_query_dataset):
    stub_query_dataset(
        equity,
        {
            "girai_country_classifications": CLASSIFICATIONS,
            "girai_scores": GIRAI,
            "oxford_ai_readiness": OXFORD,
        },
    )


def test_joins_by_iso3_and_groups_by_region(stub_query_dataset):
    _stub(stub_query_dataset)
    result = equity.regional_equity_scores(group_by="region")
    assert result["group_by"] == "region"

    by_group = {g["group"]: g for g in result["groups"]}
    africa = by_group["Africa"]
    assert africa["girai"] == {"n": 2, "mean": 35.0, "min": 30.0, "max": 40.0}
    assert africa["oxford"] == {"n": 1, "mean": 50.0, "min": 50.0, "max": 50.0}

    americas = by_group["Americas"]
    assert americas["girai"]["mean"] == 80.0
    assert americas["oxford"]["mean"] == 90.0


def test_missing_source_data_is_null_not_dropped(stub_query_dataset):
    _stub(stub_query_dataset)
    result = equity.regional_equity_scores(group_by="region")
    by_iso3 = {c["iso3"]: c for c in result["countries"]}

    # Nigeria: in classifications + GIRAI, missing from Oxford -> null, not dropped.
    assert by_iso3["NGA"]["girai"] == 30.0
    assert by_iso3["NGA"]["oxford"] is None

    # Palestine: in classifications only -> both null, but still present.
    assert by_iso3["PSE"]["girai"] is None
    assert by_iso3["PSE"]["oxford"] is None
    assert by_iso3["PSE"]["country"] == "Palestine"


def test_groups_sorted_by_girai_mean_ascending(stub_query_dataset):
    _stub(stub_query_dataset)
    result = equity.regional_equity_scores(group_by="region")
    means = [g["girai"]["mean"] for g in result["groups"]]
    non_null = [m for m in means if m is not None]
    assert non_null == sorted(non_null)


def test_group_by_is_caller_specified_column(stub_query_dataset):
    stub_query_dataset(
        equity,
        {
            "girai_country_classifications": {
                "dataset": "girai_country_classifications",
                "columns": ["iso3", "country", "wb_income_group"],
                "rows": [
                    {"iso3": "KEN", "country": "Kenya", "wb_income_group": "Lower middle income"},
                    {"iso3": "USA", "country": "United States", "wb_income_group": "High income"},
                ],
            },
            "girai_scores": GIRAI,
            "oxford_ai_readiness": OXFORD,
        },
    )
    result = equity.regional_equity_scores(group_by="wb_income_group")
    assert result["group_by"] == "wb_income_group"
    groups = {g["group"] for g in result["groups"]}
    assert groups == {"Lower middle income", "High income"}
