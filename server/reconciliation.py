"""Cross-source model name reconciliation — the single reusable join key.

model_benchmarks (Artificial Analysis), arena_agent_leaderboard (LMArena), and
openrouter_rankings (OpenRouter) each name the same model differently and share no
common id:

    model_benchmarks.name        "Claude Fable 5 (Adaptive Reasoning, Max Effort...)"
    arena_agent_leaderboard.name "Claude Fable 5 (High)"
    openrouter_rankings.model_permaslug   "anthropic/claude-fable-5"

normalize_model_name() reduces any of these to one canonical key so a single
matched-by-key join works everywhere instead of being re-implemented per tool
(scorecard.model_scorecard and any future cross-source tool all call this).

Recon status (see planning/02-name-reconciliation.md): this dev environment has no
live warehouse access, so real match-rate numbers haven't been measured yet. Per the
doc's decision gate, we're starting **extend-in-code** (this module) rather than a
crosswalk table, on the expectation that a permaslug-aware normalizer clears the
~90% bar for well-known frontier models. Run the recon queries once deployed and add
stragglers to MODEL_NAME_OVERRIDES below; if match rates come back poor, revisit the
crosswalk-table option instead.
"""

from __future__ import annotations

import re

# Exact-match overrides for normalized keys the heuristic can't reconcile on its
# own. Keys and values must already be normalized (lowercase, whitespace-collapsed,
# separators-collapsed, parenthetical-stripped) — i.e. in the same shape
# normalize_model_name() produces before consulting this map. Populate as real
# mismatches turn up when running model_scorecard against live data.
MODEL_NAME_OVERRIDES: dict[str, str] = {}


def normalize_model_name(raw_name: str | None, *, is_permaslug: bool = False) -> str:
    """Reduce a model's display name / slug / permaslug to a canonical join key.

    Steps: lowercase; for permaslugs, drop the `vendor/` prefix; strip a trailing
    parenthetical suffix (reasoning effort, mode, thinking budget — the part that
    varies most between arena and benchmark naming); collapse `-`/`_`/`.` and
    whitespace runs to single spaces; then apply MODEL_NAME_OVERRIDES for pairs
    this heuristic can't reconcile on its own.
    """
    if not raw_name:
        return ""
    name = raw_name.strip().lower()
    if is_permaslug and "/" in name:
        name = name.split("/", 1)[1]
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = re.sub(r"[-_.]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return MODEL_NAME_OVERRIDES.get(name, name)


def index_by_key(
    rows: list[dict], name_column: str, *, is_permaslug: bool = False
) -> dict[str, dict]:
    """Build {normalized_key: row} from a list of rows, dropping unkeyable rows.

    Last row wins on a duplicate key — callers that care about ordering should
    pre-sort (e.g. most-recent-first) before calling this.
    """
    index: dict[str, dict] = {}
    for row in rows:
        key = normalize_model_name(row.get(name_column), is_permaslug=is_permaslug)
        if key:
            index[key] = row
    return index
