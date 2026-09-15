"""regional_equity_scores — the deterministic core of the global-equity read (Level 2).

Pulled out of recipes/global_equity_representation.md: joining
girai_country_classifications (region/income/development reference) to
girai_scores (responsible-AI governance) + oxford_ai_readiness (government AI
readiness) by iso3, then averaging by whatever grouping dimension the caller
asks for, is a reproducible aggregation — the same category as
analysis.capability_cost_pareto — so it's a tool, not left to Claude to
recompute differently each time.

The rest of the global-equity read (SAHARA language coverage, open-vs-closed
model matching, the narrative synthesis) stays in the MD recipe: those steps
are either fuzzy name-matching (explicitly not reproducible) or interpretive,
not aggregation.
"""

from __future__ import annotations

from typing import Any

from server import retrieval
from server.stats import to_float

_COUNTRY_LIMIT = 300


def _index_by_iso3(rows: list[dict[str, Any]], value_column: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        iso3 = row.get("iso3")
        value = to_float(row.get(value_column))
        if iso3 and value is not None:
            scores[iso3] = value
    return scores


def _group_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 2) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def regional_equity_scores(group_by: str = "region") -> dict[str, Any]:
    """Join country classification + GIRAI responsibility + Oxford readiness by iso3,
    then aggregate both scores by group_by.

    group_by is any column in girai_country_classifications (e.g. region,
    subregion, developing_status, least_developed_countries, wb_income_group)
    — validated live against the warehouse schema, not hardcoded.

    GIRAI covers 135 countries and Oxford covers ~195 — the two lists don't
    match. A country missing from one source keeps a null for that source
    rather than being dropped from the read entirely, and group-level
    mean/min/max are computed only over the countries that actually have a
    score for that source (n_girai / n_oxford track how many that was, so a
    thin group isn't mistaken for a robust one).

    Returns:
        {group_by, groups: [{group, girai: {n,mean,min,max}, oxford: {...}}],
         countries: [{iso3, country, group, girai, oxford}]}
    """
    classifications = retrieval.query_dataset(
        "girai_country_classifications",
        columns=["iso3", "country", group_by],
        limit=_COUNTRY_LIMIT,
    )["rows"]
    girai_scores = _index_by_iso3(
        retrieval.query_dataset("girai_scores", columns=["iso3", "girai"], limit=_COUNTRY_LIMIT)[
            "rows"
        ],
        "girai",
    )
    oxford_scores = _index_by_iso3(
        retrieval.query_dataset(
            "oxford_ai_readiness", columns=["iso3", "total_score"], limit=_COUNTRY_LIMIT
        )["rows"],
        "total_score",
    )

    countries: list[dict[str, Any]] = []
    by_group: dict[Any, dict[str, list[float]]] = {}
    for row in classifications:
        iso3 = row.get("iso3")
        if not iso3:
            continue
        group = row.get(group_by)
        girai = girai_scores.get(iso3)
        oxford = oxford_scores.get(iso3)
        countries.append(
            {
                "iso3": iso3,
                "country": row.get("country"),
                "group": group,
                "girai": girai,
                "oxford": oxford,
            }
        )
        bucket = by_group.setdefault(group, {"girai": [], "oxford": []})
        if girai is not None:
            bucket["girai"].append(girai)
        if oxford is not None:
            bucket["oxford"].append(oxford)

    groups = [
        {
            "group": group,
            "girai": _group_stats(vals["girai"]),
            "oxford": _group_stats(vals["oxford"]),
        }
        for group, vals in by_group.items()
    ]
    groups.sort(key=lambda g: (g["girai"]["mean"] is None, g["girai"]["mean"] or 0))

    return {"group_by": group_by, "groups": groups, "countries": countries}
