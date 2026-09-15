# Recipe: global equity representation

The most mission-relevant L3 read: does frontier AI serve regions/languages equally?
Weaves SAHARA (African-language model coverage) + GIRAI (responsibility) + Oxford
(government readiness), sliced by region/income group via
`girai_country_classifications`. Pure **MD orchestration** — no dedicated Python tool.
Per `planning/00-overview.md`'s implementation model: the data pulls are grounded and
exact (`query_dataset`), the cross-source reasoning (matching a SAHARA model name to
its `model_benchmarks` open/closed status, connecting country-level readiness gaps to
language coverage gaps) is Claude's job, not a reproducibility-guaranteed join — unlike
`model_scorecard`, this narrative doesn't need a hard "same inputs → same output"
guarantee, it needs a well-grounded story.

## Steps

1. **Country reference frame.** `query_dataset('girai_country_classifications')` for
   region, income group, GDP, development status. This is the join key for everything
   country-level below — do the region/income grouping yourself from these fields
   rather than assuming any other table pre-labels it.
2. **Responsibility.** `query_dataset('girai_scores', ...)` (overall + dimension/pillar,
   most recent edition — check `girai_editions_comparison` if a 2024-vs-2026 trend adds
   something). Group by region/income group from step 1; note which
   regions/income-groups cluster low.
3. **Readiness.** `query_dataset('oxford_ai_readiness', ...)` (+ `oxford_readiness_dimensions`
   for which pillar is driving a low score). Same grouping. Look for regions that are
   low on *both* readiness and responsibility — that's a compounding gap, not two
   separate stories.
4. **Language coverage.** `query_dataset('sahara_leaderboard', ...)` for overall model
   scores, `sahara_language_scores` for per-language detail, `sahara_task_scores` if a
   specific task matters to the question asked. Identify which African languages are
   well-covered by frontier models and which aren't.
5. **Open vs. closed, for the African-language angle.** Pull `model_benchmarks`
   (`name`, `is_open_weights`, pricing) and match each SAHARA-leaderboard model to it
   by name (reuse the reasoning in `server/reconciliation.py`'s approach — lowercase,
   strip trailing parenthetical, don't force a match you're not confident in). For
   models you can confidently match, identify the best-performing **open-weight**
   model on African languages vs. the best-performing model overall, and the cost gap
   (`model_roi` or `cost_efficiency_frontier` can supply the cost side). State
   unmatched models explicitly rather than silently excluding them from the read.
6. **Synthesize.** Connect the three layers: do the regions/income-groups with weak
   government AI readiness and weak responsibility scores (steps 2–3) also have weak
   language coverage from frontier models (step 4)? Is the best African-language
   model open-weight (accessible to those regions without a large compute/API budget)
   or closed (a cost barrier compounding the readiness gap)? That intersection — not
   any single table — is the point of this tool.

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
