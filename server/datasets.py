"""The whitelist of datasets the MCP server is allowed to touch.

This is the single source of truth for which Unity Catalog tables are
queryable, and under what fully-qualified name. Allowed *columns* are
deliberately not hardcoded here: they are fetched live from
INFORMATION_SCHEMA.COLUMNS (see server/retrieval.py) so this file doesn't
drift out of sync with the warehouse schema. Table membership in DATASETS
below is still the explicit, hand-maintained gate — nothing outside this
dict is ever reachable through the server's tools.
"""

from __future__ import annotations

from dataclasses import dataclass

CATALOG = "fso_market_intelligence"
SCHEMA = "frontier_labs"


class DatasetError(Exception):
    """Base class for whitelist validation failures. Fail closed, always."""


class UnknownDatasetError(DatasetError):
    """Raised when a caller asks for a dataset that isn't in the whitelist."""

    def __init__(self, dataset: str):
        super().__init__(
            f"Unknown dataset '{dataset}'. Call list_datasets() for the "
            "set of valid dataset names."
        )
        self.dataset = dataset


@dataclass(frozen=True)
class Dataset:
    name: str
    table: str
    category: str
    description: str
    has_history: bool

    @property
    def full_name(self) -> str:
        return f"{CATALOG}.{SCHEMA}.{self.table}"


_DATASET_LIST = [
    # Hiring / frontier labs — NOT REFRESHED since ~2026-07-28, treat as stale.
    Dataset(
        name="hiring_jobs",
        table="hiring_jobs",
        category="Hiring / frontier labs",
        description=(
            "One row per individual job posting at a frontier lab (title, category, "
            "status). (NOT REFRESHED since ~2026-07-28 — treat as stale.)"
        ),
        has_history=False,
    ),
    Dataset(
        name="hiring_weekly",
        table="hiring_weekly",
        category="Hiring / frontier labs",
        description=(
            "One row per company per week: open/new/removed job counts, incl. "
            "social-impact roles. (NOT REFRESHED since ~2026-07-28 — treat as stale.)"
        ),
        has_history=True,
    ),
    Dataset(
        name="hiring_weekly_breakdown",
        table="hiring_weekly_breakdown",
        category="Hiring / frontier labs",
        description=(
            "One row per company per week per category/sub_area/theme/vertical: job "
            "counts at that grain. (NOT REFRESHED since ~2026-07-28 — treat as stale.)"
        ),
        has_history=True,
    ),
    # Model capability & perception
    Dataset(
        name="model_benchmarks",
        table="model_benchmarks",
        category="Model capability & perception",
        description=(
            "One row per frontier model, snapshotting Artificial Analysis's current "
            "methodology: capability evals (e.g. intelligence_index, gpqa_diamond, hle), "
            "pricing (price_blended_3to1, cost_per_task_total), is_open_weights, "
            "release_date. Overwritten daily in place — no history retained; use "
            "release_date as the time axis for any over-time framing, not this table's "
            "load date."
        ),
        has_history=False,
    ),
    Dataset(
        name="arena_agent_leaderboard",
        table="arena_agent_leaderboard",
        category="Model capability & perception",
        description=(
            "One row per agentic model, current LMArena Agent Arena overall score/rank "
            "(stated preference — what users say they prefer)."
        ),
        has_history=False,
    ),
    Dataset(
        name="arena_agent_signals",
        table="arena_agent_signals",
        category="Model capability & perception",
        description=(
            "One row per agentic model per sub-dimension (steerability, tool "
            "hallucination, etc.) from LMArena."
        ),
        has_history=False,
    ),
    # Compute cost
    Dataset(
        name="ornn_gpu_prices",
        table="ornn_gpu_prices",
        category="Compute cost",
        description=(
            "One row per (date, gpu_name): daily settled GPU $/hr spot index (Ornn OCPI, "
            "free tier). Tracks 5 accelerators only (4 datacenter + 1 consumer) — not the "
            "full hardware market. Effective upsert keyed (date, gpu_name); ~3mo seed + "
            "daily append."
        ),
        has_history=True,
    ),
    # Model adoption / usage
    Dataset(
        name="openrouter_rankings",
        table="openrouter_rankings",
        category="Model adoption / usage",
        description=(
            "One row per (date, model_permaslug): daily top models by real token volume "
            "on OpenRouter (revealed preference — what people actually run). Full history "
            "back to 2025-01-01, weekly refresh. is_aggregate=true rows are author/family "
            "rollups, not single models — filter to is_aggregate=false for per-model reads."
        ),
        has_history=True,
    ),
    # African-NLP (SAHARA)
    Dataset(
        name="sahara_leaderboard",
        table="sahara_leaderboard",
        category="African-NLP (SAHARA)",
        description=(
            "One row per model: overall and per-cluster scores on the SAHARA "
            "African-NLP benchmark (point-in-time snapshot, not appended over time)."
        ),
        has_history=False,
    ),
    Dataset(
        name="sahara_scores",
        table="sahara_scores",
        category="African-NLP (SAHARA)",
        description="One row per model/task/language: granular SAHARA scores at that grain.",
        has_history=False,
    ),
    Dataset(
        name="sahara_task_scores",
        table="sahara_task_scores",
        category="African-NLP (SAHARA)",
        description="One row per model per SAHARA task: task-level scores.",
        has_history=False,
    ),
    Dataset(
        name="sahara_language_scores",
        table="sahara_language_scores",
        category="African-NLP (SAHARA)",
        description="One row per model per African language: language-level scores.",
        has_history=False,
    ),
    # Government AI readiness (Oxford)
    Dataset(
        name="oxford_ai_readiness",
        table="oxford_ai_readiness",
        category="Government AI readiness (Oxford)",
        description=(
            "One row per country: Oxford Insights Government AI Readiness Index overall "
            "score + pillar scores (point-in-time index edition)."
        ),
        has_history=False,
    ),
    Dataset(
        name="oxford_readiness_dimensions",
        table="oxford_readiness_dimensions",
        category="Government AI readiness (Oxford)",
        description="One row per country per readiness sub-dimension: Oxford dimension scores.",
        has_history=False,
    ),
    # Responsible AI (GIRAI) + country reference
    Dataset(
        name="girai_scores",
        table="girai_scores",
        category="Responsible AI (GIRAI)",
        description=(
            "One row per country: overall (girai, post-URAI-penalty), dimension, and "
            "pillar GIRAI scores/ranking for the current edition (2026). Overwritten each "
            "load with a single hardcoded edition — does not accumulate prior editions; "
            "use girai_editions_comparison for 2024-vs-2026."
        ),
        has_history=False,
    ),
    Dataset(
        name="girai_pillar_dimension_scores",
        table="girai_pillar_dimension_scores",
        category="Responsible AI (GIRAI)",
        description=(
            "One row per country per pillar per dimension: GIRAI scores at that grain for "
            "the current edition (2026). Overwritten each load, single edition only."
        ),
        has_history=False,
    ),
    Dataset(
        name="girai_indicators",
        table="girai_indicators",
        category="Responsible AI (GIRAI)",
        description=(
            "One row per country per indicator: GIRAI indicator-level scores for the "
            "current edition (2026). Overwritten each load, single edition only."
        ),
        has_history=False,
    ),
    Dataset(
        name="girai_editions_comparison",
        table="girai_editions_comparison",
        category="Responsible AI (GIRAI)",
        description=(
            "One row per country per indicator: status/initiatives compared 2024 vs 2026 "
            "side by side (already a two-edition comparison, not an appended series)."
        ),
        has_history=True,
    ),
    Dataset(
        name="girai_country_classifications",
        table="girai_country_classifications",
        category="Responsible AI (GIRAI)",
        description=(
            "One row per country: reference data (region, income group, GDP, development "
            "status) used to slice the other GIRAI/Oxford/SAHARA tables by geography."
        ),
        has_history=False,
    ),
]

DATASETS: dict[str, Dataset] = {d.name: d for d in _DATASET_LIST}


def get_dataset(name: str) -> Dataset:
    """Look up a dataset by name, failing closed on anything not whitelisted."""
    try:
        return DATASETS[name]
    except KeyError:
        raise UnknownDatasetError(name) from None


def dataset_names() -> list[str]:
    return list(DATASETS.keys())
