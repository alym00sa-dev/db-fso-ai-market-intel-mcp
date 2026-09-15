# Recipe: global equity representation

The most mission-relevant L3 read: does frontier AI serve regions/languages equally?
Weaves SAHARA (African-language model coverage) + GIRAI (responsibility) + Oxford
(government readiness), sliced by region/income group. **Hybrid MD orchestration**:
the region/income-group readiness+responsibility join is deterministic and pulled out
as `server/equity.py`'s `regional_equity_scores` tool (Level 2 — same category as
`capability_cost_pareto`); everything else (language coverage, matching SAHARA models
to `model_benchmarks` open/closed status, and the narrative synthesis) stays Claude's
job at query time, since the SAHARA name-match is explicitly best-effort, not a
reproducibility-guaranteed join like `model_scorecard`'s reconciliation.

## Steps

1. **Readiness + responsibility, by region/income group — call `regional_equity_scores(group_by=...)`.**
   This tool does the deterministic part: joins `girai_country_classifications`
   (region/income/development reference) to `girai_scores` (responsibility) and
   `oxford_ai_readiness` (readiness) by `iso3`, and returns per-group mean/min/max for
   both scores (`group_by` can be `region`, `subregion`, `developing_status`, or
   `wb_income_group` — pass whichever the question is actually about). Look for groups
   that are low on *both* `girai` and `oxford` — that's a compounding gap, not two
   separate stories. Note `n_girai`/`n_oxford` per group before trusting a mean — GIRAI
   covers 135 countries, Oxford ~195, so a thin group (small n) shouldn't be read with
   the same confidence as a well-populated one. For a specific country's dimension-level
   detail (which pillar is driving a low score), drop down to `girai_pillar_dimension_scores`
   / `oxford_readiness_dimensions` directly via `query_dataset`.
2. **Language coverage.** `query_dataset('sahara_leaderboard', ...)` for overall model
   scores, `sahara_language_scores` for per-language detail, `sahara_task_scores` if a
   specific task matters to the question asked. Identify which African languages are
   well-covered by frontier models and which aren't.
3. **Open vs. closed, for the African-language angle.** Pull `model_benchmarks`
   (`name`, `is_open_weights`, pricing) and match each SAHARA-leaderboard model to it
   by name (reuse the reasoning in `server/reconciliation.py`'s approach — lowercase,
   strip trailing parenthetical, don't force a match you're not confident in). For
   models you can confidently match, identify the best-performing **open-weight**
   model on African languages vs. the best-performing model overall, and the cost gap
   (`capability_cost_pareto` can supply the cost side). State unmatched models
   explicitly rather than silently excluding them from the read.
4. **Synthesize.** Connect the layers: do the regions/income-groups `regional_equity_scores`
   flagged as weak on readiness+responsibility (step 1) also have weak language coverage
   from frontier models (step 2)? Is the best African-language model open-weight
   (accessible to those regions without a large compute/API budget) or closed (a cost
   barrier compounding the readiness gap)? That intersection — not any single table —
   is the point of this read.

## Guardrails
- Every country-level claim must trace to a specific `girai_*` / `oxford_*` row —
  don't extrapolate a region's story from one country in it.
- Name-matching SAHARA models to `model_benchmarks` is best-effort narrative reasoning,
  not a guaranteed join (unlike `model_scorecard`'s reconciliation). Say "likely the
  same model as X" rather than asserting it silently if the match isn't clean, and
  don't drop unmatched models from the read without saying so.
- Hiring data isn't in this server's whitelist and is out of scope for this recipe.
- If a requested country/region has no rows in one of the four table families, say
  "no data" for that layer rather than omitting the country from the whole read.
