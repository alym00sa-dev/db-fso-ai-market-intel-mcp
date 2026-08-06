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
    # Hiring / frontier labs
    Dataset(
        name="hiring_jobs",
        table="hiring_jobs",
        category="Hiring / frontier labs",
        description="Individual job postings at frontier labs (title, category, status).",
        has_history=False,
    ),
    Dataset(
        name="hiring_weekly",
        table="hiring_weekly",
        category="Hiring / frontier labs",
        description="Weekly open/new/removed jobs per company, incl. social-impact roles.",
        has_history=True,
    ),
    Dataset(
        name="hiring_weekly_breakdown",
        table="hiring_weekly_breakdown",
        category="Hiring / frontier labs",
        description="Weekly job counts per company by category/sub_area/theme/vertical.",
        has_history=True,
    ),
    # Model capability & perception
    Dataset(
        name="model_benchmarks",
        table="model_benchmarks",
        category="Model capability & perception",
        description="Benchmark results + metadata for frontier models (composite scores, pricing).",
        has_history=True,
    ),
    Dataset(
        name="arena_agent_leaderboard",
        table="arena_agent_leaderboard",
        category="Model capability & perception",
        description="LMArena Agent Arena overall scores/ranks for agentic models.",
        has_history=False,
    ),
    Dataset(
        name="arena_agent_signals",
        table="arena_agent_signals",
        category="Model capability & perception",
        description="LMArena sub-dimension scores (steerability, tool hallucination, etc.).",
        has_history=False,
    ),
    # African-NLP (SAHARA)
    Dataset(
        name="sahara_leaderboard",
        table="sahara_leaderboard",
        category="African-NLP (SAHARA)",
        description="Overall and cluster scores on the SAHARA African-NLP benchmark.",
        has_history=False,
    ),
    Dataset(
        name="sahara_scores",
        table="sahara_scores",
        category="African-NLP (SAHARA)",
        description="Granular SAHARA scores by model, task, and language.",
        has_history=False,
    ),
    Dataset(
        name="sahara_task_scores",
        table="sahara_task_scores",
        category="African-NLP (SAHARA)",
        description="Model scores for specific SAHARA tasks.",
        has_history=False,
    ),
    Dataset(
        name="sahara_language_scores",
        table="sahara_language_scores",
        category="African-NLP (SAHARA)",
        description="Model scores for individual African languages.",
        has_history=False,
    ),
    # Government AI readiness (Oxford)
    Dataset(
        name="oxford_ai_readiness",
        table="oxford_ai_readiness",
        category="Government AI readiness (Oxford)",
        description="Oxford Insights Government AI Readiness Index scores/pillars per country.",
        has_history=False,
    ),
    Dataset(
        name="oxford_readiness_dimensions",
        table="oxford_readiness_dimensions",
        category="Government AI readiness (Oxford)",
        description="Oxford readiness sub-dimension scores per country.",
        has_history=False,
    ),
    # Responsible AI (GIRAI) + country reference
    Dataset(
        name="girai_scores",
        table="girai_scores",
        category="Responsible AI (GIRAI)",
        description="Overall, dimension, and pillar GIRAI scores/rankings per country.",
        has_history=True,
    ),
    Dataset(
        name="girai_pillar_dimension_scores",
        table="girai_pillar_dimension_scores",
        category="Responsible AI (GIRAI)",
        description="GIRAI scores by country, pillar, and dimension.",
        has_history=True,
    ),
    Dataset(
        name="girai_indicators",
        table="girai_indicators",
        category="Responsible AI (GIRAI)",
        description="GIRAI indicator-level scores per country and edition.",
        has_history=True,
    ),
    Dataset(
        name="girai_editions_comparison",
        table="girai_editions_comparison",
        category="Responsible AI (GIRAI)",
        description="GIRAI indicator status/initiatives compared 2024 vs 2026.",
        has_history=True,
    ),
    Dataset(
        name="girai_country_classifications",
        table="girai_country_classifications",
        category="Responsible AI (GIRAI)",
        description="Country reference data: region, income group, GDP, development status.",
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
