"""Cross-source model name reconciliation — the single reusable join key.

model_benchmarks (Artificial Analysis), arena_agent_leaderboard (LMArena), and
openrouter_rankings (OpenRouter) each name the same model differently and share no
common id:

    model_benchmarks.name        "Claude Fable 5 (Adaptive Reasoning, Max Effort...)"
    arena_agent_leaderboard.name "Claude Fable 5 (High)"
    openrouter_rankings.model_permaslug   "anthropic/claude-fable-5"

tokenize_model_name() / normalize_model_name() / best_token_match() reduce any of
these to one canonical key or match so matching logic is written once instead of
being re-implemented per tool (compute_fit's model lookup and
scorecard.model_scorecard's cross-source signals all call this).

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
from typing import TypeVar

# Exact-match overrides for normalized keys the heuristic can't reconcile on its
# own. Keys and values must already be normalized (lowercase, sorted-token form) —
# i.e. in the same shape normalize_model_name() produces before consulting this
# map. Populate as real mismatches turn up when running model_scorecard /
# compute_fit against live data.
MODEL_NAME_OVERRIDES: dict[str, str] = {}

_VERSION_MERGE_MAX_LEN = 2  # "3", "3" -> "3.3"; won't touch e.g. "70b" (has a letter)


def tokenize_model_name(raw_name: str | None, *, is_permaslug: bool = False) -> list[str]:
    """Split a model name/slug/permaslug into tokens, preserving decimal versions.

    Lowercase; for permaslugs, drop the `vendor/` prefix; strip a trailing
    parenthetical suffix (reasoning effort, mode, thinking budget); split on
    whitespace/hyphens/underscores/parens/commas — but **not dots**, so a
    decimal version like "3.3" stays one token instead of fragmenting into a
    bare "3" that collides with every other point release ("3.1", "3.5", ...).
    Then merge adjacent short all-digit tokens back into a single "X.Y" token,
    so a hyphen-written version in a slug (e.g. "llama-3-3-70b") still produces
    "3.3" instead of two separate "3" tokens — confirmed necessary against live
    data, where naive matching on "llama 3.3 70b" surfaced several Llama 3.1
    rows because both "3.3" and "3.1" degraded to a shared "3".
    """
    if not raw_name:
        return []
    name = raw_name.strip().lower()
    if is_permaslug and "/" in name:
        name = name.split("/", 1)[1]
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    raw_tokens = [t for t in re.split(r"[\s\-_(),]+", name) if t]

    tokens: list[str] = []
    i = 0
    while i < len(raw_tokens):
        cur = raw_tokens[i]
        nxt = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else None
        if (
            nxt is not None
            and cur.isdigit()
            and nxt.isdigit()
            and len(cur) <= _VERSION_MERGE_MAX_LEN
            and len(nxt) <= _VERSION_MERGE_MAX_LEN
        ):
            tokens.append(f"{cur}.{nxt}")
            i += 2
        else:
            tokens.append(cur)
            i += 1
    return tokens


def normalize_model_name(raw_name: str | None, *, is_permaslug: bool = False) -> str:
    """Reduce a model's display name / slug / permaslug to a canonical join key.

    Tokenizes (see tokenize_model_name — decimal versions preserved), sorts the
    tokens — confirmed against live data that the same model can appear in a
    different word order per source (e.g. AA's "Llama 3.3 Instruct 70B" vs. a
    permaslug-derived "llama 3.3 70b instruct"), which breaks plain ordered-
    string equality — then applies MODEL_NAME_OVERRIDES for pairs this
    heuristic still can't reconcile.
    """
    tokens = tokenize_model_name(raw_name, is_permaslug=is_permaslug)
    if not tokens:
        return ""
    name = " ".join(sorted(tokens))
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


_T = TypeVar("_T")


def best_token_match(
    query: str, candidates: list[tuple[str, _T]], *, is_permaslug: bool = False
) -> tuple[_T, str, list[str]] | None:
    """Find the best candidate whose tokens are mutually compatible with the query's.

    `candidates` is a list of (display_name, payload) pairs. A candidate
    qualifies if one token set contains the other — **bidirectional**, since
    for cross-source matching (e.g. querying arena/OpenRouter with a name
    resolved from model_benchmarks) either side can be the more verbose one:
    AA might say "Llama 3.3 Instruct 70B" while arena says "Llama 3.3 70B
    (High)" — neither is a strict superset of the other's full name, but they
    agree on every token they share. Candidates are ranked by the symmetric
    difference (fewest mismatched tokens in either direction), ties broken by
    name. Returns (payload, matched_name, other_candidate_names[:4]) for the
    best match, or None if nothing qualifies. Never auto-resolves ambiguity
    beyond "best match wins" — callers that need to distinguish a clean single
    match from a near-tie should inspect `other_candidates`.
    """
    query_tokens = set(tokenize_model_name(query))
    if not query_tokens:
        return None
    scored: list[tuple[int, str, _T]] = []
    for name, payload in candidates:
        row_tokens = set(tokenize_model_name(name, is_permaslug=is_permaslug))
        if not row_tokens:
            continue
        if query_tokens.issubset(row_tokens) or row_tokens.issubset(query_tokens):
            scored.append((len(query_tokens.symmetric_difference(row_tokens)), name, payload))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    best = scored[0]
    others = [s[1] for s in scored[1:5]]
    return best[2], best[1], others
