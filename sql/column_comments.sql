-- Table + column comments for all 19 whitelisted datasets in
-- fso_market_intelligence.frontier_labs (server/datasets.py). Run against the
-- warehouse with a role that has MODIFY on these tables.
--
-- Source of truth: the loader notebooks in databricks/frontier_labs/ (read-only
-- reference — nothing there was or should be modified) plus
-- databricks/frontier_labs/girai_genie_instructions.md, which already has
-- ready-written column descriptions for the GIRAI tables. Not sourced from
-- databricks/frontier_labs/genie_context.md's model_benchmarks section, which
-- documents the OLD pre-premium schema and is stale for that one table (its
-- other tables' descriptions are still accurate and were cross-checked).
--
-- Safe to run statement-by-statement; a few are flagged below as best-effort
-- because the loader notebook doesn't explicitly confirm them (only inferred
-- from prose docs) — if one of those errors with "column does not exist",
-- skip it rather than treating it as a script failure.
--
-- model_benchmarks caveat: refresh_benchmarks.ipynb pulls evaluation + performance
-- metrics DYNAMICALLY from the Artificial Analysis Pro API (whatever keys are
-- present that day), so this file only comments the STATIC columns the notebook
-- hardcodes. Run
--   SELECT column_name FROM fso_market_intelligence.information_schema.columns
--   WHERE table_schema = 'frontier_labs' AND table_name = 'model_benchmarks'
-- to see the live dynamic eval/perf columns before commenting those individually.


-- ============================================================
-- model_benchmarks — Artificial Analysis Pro, one row per model, OVERWRITTEN
-- DAILY (no history retained — use release_date as the time axis).
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.model_benchmarks IS
    'Frontier model capability/pricing snapshot from Artificial Analysis Pro (refresh_benchmarks.ipynb). Overwritten daily — no history retained; use release_date as the time axis, not captured_at.';

COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.id IS 'Artificial Analysis internal model id.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.name IS 'Model display name (join key used across this schema''s Python tools).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.slug IS 'Artificial Analysis URL-safe model slug.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.org IS 'Model creator / organization name (NOT "organization").';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.country IS 'Model creator''s country.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.release_date IS 'Model''s public release date — the time axis for any "over time" question on this table (the table itself has no history).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.is_open_weights IS 'True if model weights are open/downloadable; false = closed/API-only (gates self-host in compute_fit/model_scorecard).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.reasoning_model IS 'True if this is a reasoning-tuned model variant.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.params_total_b IS 'Total parameter count, in billions. Used by compute_fit for VRAM sizing.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.params_active_b IS 'Active parameter count per token, in billions (MoE models activate a subset of total params; equals params_total_b for dense models).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.context_window_tokens IS 'Maximum context window, in tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_input IS 'Price per 1M input tokens, USD.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_output IS 'Price per 1M output tokens, USD.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_blended_3to1 IS 'Blended $/1M tokens at a 3:1 input:output ratio — the default cost column for model_roi/cost_efficiency_frontier.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_blended_7to2to1 IS 'Blended $/1M tokens at Artificial Analysis''s 7:2:1 input:output:reasoning ratio.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_cache_hit IS 'Price per 1M cache-hit input tokens, USD.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.price_cache_write IS 'Price per 1M cache-write input tokens, USD.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.cost_per_task_total IS 'Dollar cost to run Artificial Analysis''s Intelligence Index eval suite as one "task" — the default cost column for cost_efficiency_frontier.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.cost_per_task_input IS 'Portion of cost_per_task_total attributable to input tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.cost_per_task_reasoning IS 'Portion of cost_per_task_total attributable to reasoning tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.cost_per_task_answer IS 'Portion of cost_per_task_total attributable to answer/output tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.eval_suite_cost_total IS 'Total dollar cost of running the full Artificial Analysis eval suite for this model.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.eval_suite_cost_input IS 'Portion of eval_suite_cost_total attributable to input tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.eval_suite_cost_reasoning IS 'Portion of eval_suite_cost_total attributable to reasoning tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.eval_suite_cost_answer IS 'Portion of eval_suite_cost_total attributable to answer/output tokens.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.tok_input IS 'Input tokens consumed running the eval suite.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.tok_output IS 'Output tokens consumed running the eval suite.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.tok_reasoning IS 'Reasoning tokens consumed running the eval suite.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.tok_answer IS 'Answer tokens consumed running the eval suite.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.captured_at IS 'Date this row was last refreshed. Table is overwritten daily — this is a load date, not a history key.';
-- Best-effort, unconfirmed live (evaluations are pulled dynamically — see file header):
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.intelligence_index IS 'Artificial Analysis composite capability score (0-100, higher = more capable). Dynamic eval column — verify it exists live before relying on the name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.model_benchmarks.hle IS 'Humanity''s Last Exam benchmark score. Dynamic eval column — verify it exists live before relying on the name.';


-- ============================================================
-- arena_agent_leaderboard — LMArena agent Elo leaderboard, full OVERWRITE monthly.
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.arena_agent_leaderboard IS
    'LMArena Agent Arena overall crowd-Elo ranking, pulled from the official HF dataset (load_arena_agent.ipynb). Full overwrite each run (monthly).';

COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.model IS 'Model display name. Renamed from the raw HF dataset''s model_name on load — NOT "model_name" in this table.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.score IS 'LMArena Elo-style agent score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.rank IS 'Rank by score on this leaderboard (1 = highest).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.ci_lower IS 'Lower bound of score''s confidence interval. Renamed from score_ci_lower on load.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.ci_upper IS 'Upper bound of score''s confidence interval. Renamed from score_ci_upper on load.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.publish_date IS 'Date LMArena published this leaderboard snapshot. Renamed from leaderboard_publish_date on load.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.captured_at IS 'Date this row was loaded into the warehouse.';
-- Best-effort, not explicitly confirmed in load_arena_agent.ipynb (only in genie_context.md prose) — skip if missing:
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.organization IS 'Model''s organization/lab, per the raw HF dataset. Not renamed by the loader — verify it exists live.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_leaderboard.open_weight IS 'True if open-weight, per the raw HF dataset. Not renamed by the loader — verify it exists live.';


-- ============================================================
-- arena_agent_signals — the 5 agent sub-dimensions, full OVERWRITE monthly.
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.arena_agent_signals IS
    'LMArena agent sub-dimension scores (steerability, tool hallucination, task outcome, praise/complaint, bash recovery). Same source/cadence as arena_agent_leaderboard.';

COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.model IS 'Model display name. Renamed from model_name on load.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.signal IS 'Which agent sub-dimension this row measures: task_outcome, steerability, tool_hallucination, praise_complaint, or bash_recovery_steps.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.score IS 'LMArena score for this model on this sub-dimension.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.rank IS 'Rank by score within this sub-dimension.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.ci_lower IS 'Lower bound of score''s confidence interval.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.ci_upper IS 'Upper bound of score''s confidence interval.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.publish_date IS 'Date LMArena published this snapshot.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.arena_agent_signals.captured_at IS 'Date this row was loaded into the warehouse.';


-- ============================================================
-- ornn_gpu_prices — daily settled GPU $/hr index, upserted keyed (date, gpu_name).
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.ornn_gpu_prices IS
    'Daily settled GPU $/hr spot price index (Ornn OCPI, free tier), 5 tracked accelerators only — see server/compute_fit.py GPU_SPECS. Upserted keyed (date, gpu_name); ~3mo seed + daily append.';

COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.date IS 'Date this settled price applies to.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.gpu_name IS 'Accelerator name as reported by Ornn (one of: B200, H200, H100 SXM, A100 SXM4, RTX 5090).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.index_value_usd_hr IS 'Settled GPU $/hr spot price index for this GPU on this date.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.source IS 'Upstream source for this row: ornn_history (seed CSV) or ornn_daily (live pull).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.source_updated_at IS 'Timestamp Ornn last updated this value.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.ornn_gpu_prices.captured_at IS 'Timestamp this row was captured into the warehouse.';


-- ============================================================
-- openrouter_rankings — daily top-models-by-token-usage, full history from 2025-01-01,
-- full OVERWRITE weekly (OpenRouter re-serves its whole history, so overwrite is safe).
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.openrouter_rankings IS
    'Daily top models by real token usage on OpenRouter (revealed preference), full history back to 2025-01-01. Full overwrite weekly.';

COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.date IS 'Date of this daily ranking snapshot.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.model_permaslug IS 'OpenRouter permaslug, formatted "<vendor>/<model-slug>" (e.g. anthropic/claude-3.5-sonnet). Does not match model_benchmarks.name/slug directly — see server/reconciliation.py. The aggregate "other" row has no "/" and is flagged via is_aggregate instead.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.total_tokens IS 'Total tokens processed for this model on this date (revealed usage volume) — the ranking key.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.is_aggregate IS 'True for the single author/family rollup "other" row (no vendor prefix) — filter to false for per-model analysis.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.period IS 'Ranking period this row covers (currently always "day").';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.rank IS 'Rank by total_tokens on this date, computed across ALL rows for the date INCLUDING the is_aggregate row — so ranks among is_aggregate=false rows can have a gap where the aggregate row landed. Not a precise contiguous 1..N after filtering.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.openrouter_rankings.captured_at IS 'Timestamp this row was loaded into the warehouse.';


-- ============================================================
-- SAHARA — African-language NLP benchmark (UBC-NLP), full OVERWRITE monthly.
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.sahara_leaderboard IS
    'One row per model: overall SAHARA score (mean task score) + per-cluster mean, derived from sahara_task_scores. Full overwrite monthly.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_leaderboard.model IS 'Model name (leading medal/emoji markers stripped by the loader).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_leaderboard.overall_score IS 'Mean SAHARA task score across all evaluated tasks.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_leaderboard.tasks_evaluated IS 'Number of distinct SAHARA tasks this model was scored on.';
-- Per-cluster columns are pivoted dynamically (one per cluster actually present, e.g.
-- text_classification / text_generation / mccr / token_level) — check describe_dataset
-- for the live set before commenting a specific cluster column.

COMMENT ON TABLE fso_market_intelligence.frontier_labs.sahara_scores IS
    'Atomic SAHARA scores: one row per model x task x language. Full overwrite monthly.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.model IS 'Model name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.task IS 'SAHARA task id (e.g. sentiment, ner, mt_eng2xx).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.cluster IS 'Task cluster grouping: Text Classification, Text Generation, MCCR, or Token-Level.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.language IS 'Language this score is for (one of SAHARA''s 517 covered languages).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.score IS 'Model''s score for this task + language.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_scores.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.sahara_task_scores IS
    'One row per model x task: the task-level score (mean across languages). Full overwrite monthly.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_task_scores.model IS 'Model name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_task_scores.task IS 'SAHARA task id.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_task_scores.cluster IS 'Task cluster grouping: Text Classification, Text Generation, MCCR, or Token-Level.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_task_scores.task_score IS 'Model''s aggregate score for this task, across all languages.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_task_scores.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.sahara_language_scores IS
    'One row per model x language: language proficiency score (mean across tasks). Full overwrite monthly.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_language_scores.model IS 'Model name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_language_scores.language IS 'Language this score is for.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_language_scores.language_score IS 'Model''s aggregate proficiency score for this language, across all tasks.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.sahara_language_scores.captured_at IS 'Date this row was loaded into the warehouse.';


-- ============================================================
-- Oxford Government AI Readiness Index 2025 — country-level annual index, OVERWRITE.
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.oxford_ai_readiness IS
    'Oxford Insights Government AI Readiness Index 2025: one row per country, overall score + 6 pillars. Government readiness/capability — distinct from GIRAI (responsible-AI governance). Overwrite each run; re-run yearly with the new workbook.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.iso3 IS 'ISO3 country code — join key across country-level tables.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.rank IS 'Rank by total_score, 1 = highest readiness.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.total_score IS 'Overall Oxford Government AI Readiness Index score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.policy_capacity IS 'Policy Capacity pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.ai_infrastructure IS 'AI Infrastructure pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.governance IS 'Governance pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.public_sector_adoption IS 'Public Sector Adoption pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.development_diffusion IS 'Development & Diffusion pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.resilience IS 'Resilience pillar score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.year IS 'Index edition year (2025). Static edition — re-run yearly, not appended.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_ai_readiness.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.oxford_readiness_dimensions IS
    'Long-format detail behind oxford_ai_readiness: one row per country x readiness sub-dimension (14 dimensions under the 6 pillars). Overwrite each run.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.iso3 IS 'ISO3 country code — join key.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.dimension IS 'One of the 14 Oxford readiness sub-dimensions.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.score IS 'Country''s score on this dimension.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.year IS 'Index edition year (2025).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.oxford_readiness_dimensions.captured_at IS 'Date this row was loaded into the warehouse.';


-- ============================================================
-- GIRAI — Global Index on Responsible AI, 2026 edition. Column descriptions sourced
-- from databricks/frontier_labs/girai_genie_instructions.md (written for this exact
-- purpose). girai_scores/girai_indicators/girai_pillar_dimension_scores are OVERWRITTEN
-- each run with a single hardcoded edition (2026) — they do NOT accumulate history,
-- unlike girai_editions_comparison which bakes in a fixed 2024-vs-2026 comparison from
-- its own source spreadsheet. See server/datasets.py has_history flags.
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.girai_scores IS
    'Global Index on Responsible AI (GIRAI) 2026: one row per country — overall/dimension/pillar scores. Government RESPONSIBLE-AI GOVERNANCE — distinct from Oxford (readiness/capability). Overwritten each run with the current edition only; use girai_editions_comparison for 2024-vs-2026.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.ranking IS 'GIRAI 2026 rank, 1 = highest girai score.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.iso3 IS 'ISO3 country code — join key across country-level tables.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.region IS 'Region label as shipped in the GIRAI scores workbook (see girai_country_classifications for the fuller region/subregion/income breakdown).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.girai IS 'Overall score (0-100) AFTER the URAI penalty — the score to rank countries on, not girai_raw.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.girai_raw IS 'Overall score (0-100) BEFORE the URAI penalty.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.urai_penalty IS 'Multiplicative penalty factor for documented use of unacceptable-risk AI systems: girai = girai_raw x urai_penalty. 1.0 = no penalty.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.dim_inclusion_diversity IS 'Inclusion & Diversity dimension score (0-100).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.dim_ethics_sustainability IS 'Ethics & Sustainability dimension score (0-100).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.dim_labour_skills IS 'Labour & Skills dimension score (0-100).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.dim_trust_safety IS 'Trust & Safety dimension score (0-100).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.dim_ai_public_service IS 'AI Use in Public Service dimension score (0-100).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.pillar_ai_policy IS 'AI Policy pillar score (0-100) — government frameworks.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.pillar_cso_engagement IS 'CSO Engagement pillar score (0-100) — civil-society involvement.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.pillar_enabling_conditions IS 'Enabling Conditions pillar score (0-100) — underlying infrastructure/rights (e.g. rule of law, data protection).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.edition IS 'GIRAI edition year (currently always 2026 — this table is overwritten per edition, not accumulated).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_scores.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.girai_indicators IS
    'GIRAI long-format detail: one row per country x thematic indicator (38 indicators, e.g. Data Protection and Privacy, Gender Equality, Cybersecurity). Overwritten each run, current edition only.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.iso3 IS 'ISO3 country code — join key.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.indicator IS 'Thematic indicator name (one of 38, rolling up into the 5 dimensions).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.score IS 'Country''s score (0-100) on this indicator.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.edition IS 'GIRAI edition year (currently always 2026).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_indicators.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores IS
    'GIRAI long-format detail: one row per country x pillar x dimension. Use for pillar-within-dimension questions (e.g. strong on AI Policy but weak on CSO Engagement, within Trust & Safety). Overwritten each run, current edition only.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.iso3 IS 'ISO3 country code — join key.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.pillar IS 'One of: AI Policy, CSO Engagement, Enabling Conditions.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.dimension IS 'One of the 5 GIRAI dimensions (Inclusion & Diversity, Ethics & Sustainability, Labour & Skills, Trust & Safety, AI Use in Public Service).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.score IS 'Country''s score (0-100) for this pillar within this dimension.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.edition IS 'GIRAI edition year (currently always 2026).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_pillar_dimension_scores.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.girai_editions_comparison IS
    'GIRAI "what changed" surface: one row per country x indicator, framework status/title/link and initiative/CSO-engagement existence flags for BOTH the 2024 and 2026 editions side by side. Unlike the other girai_* tables, this genuinely spans two editions (baked into its own source spreadsheet, not appended over time).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.iso3 IS 'ISO3 country code — join key.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.indicator IS 'Thematic indicator name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.dimension IS 'GIRAI dimension this indicator rolls up into.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_status_2024 IS 'Framework status for this indicator in the 2024 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_title_2024 IS 'Framework title for this indicator in the 2024 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_link_2024 IS 'Framework URL for this indicator in the 2024 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_status_2026 IS 'Framework status for this indicator in the 2026 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_title_2026 IS 'Framework title for this indicator in the 2026 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.fr_link_2026 IS 'Framework URL for this indicator in the 2026 edition.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.init_existence_2024 IS 'Whether an initiative existed for this indicator in the 2024 edition (Yes/No).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.init_existence_2026 IS 'Whether an initiative existed for this indicator in the 2026 edition (Yes/No).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.cso_existence_2024 IS 'Whether CSO (civil-society organization) engagement existed for this indicator in the 2024 edition (Yes/No).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.cso_existence_2026 IS 'Whether CSO engagement existed for this indicator in the 2026 edition (Yes/No).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.edition IS 'Load edition tag (2026) — NOTE: this column labels the LOAD, not which of the two compared editions a given row''s data is from; see the fr_*_2024/2026 and *_existence_2024/2026 columns for that.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_editions_comparison.captured_at IS 'Date this row was loaded into the warehouse.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.girai_country_classifications IS
    'Country reference data for slicing GIRAI/Oxford/SAHARA by geography or income: region, subregion, developing status, LDC flag, World Bank income group, GDP per capita (PPP). 138 rows (135 GIRAI-scored countries + 3 reference-only territories) — join on iso3 so the extras never surface uninvited.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.iso3 IS 'ISO3 country code — join key across country-level tables.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.country IS 'Country name.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.region IS 'Region.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.subregion IS 'Subregion.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.developing_status IS 'Developed / Developing classification.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.least_developed_countries IS 'Least Developed Country (LDC) flag.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.wb_income_group IS 'World Bank income group classification.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.gdp_per_capita_ppp IS 'GDP per capita, PPP-adjusted.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.edition IS 'Load edition tag (2026).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.girai_country_classifications.captured_at IS 'Date this row was loaded into the warehouse.';


-- ============================================================
-- Hiring @ Frontier Labs — NOT REFRESHED since ~2026-07-28, treat as stale
-- (see server/datasets.py descriptions). Columns confirmed against
-- load_hiring.ipynb, the current authoritative loader (supersedes the older
-- seed_frontier_labs.ipynb, which lacks the department/title/url/source/
-- description fields hiring_jobs actually carries).
-- ============================================================
COMMENT ON TABLE fso_market_intelligence.frontier_labs.hiring_jobs IS
    'Cumulative hiring ledger, one row per job posting (keeps closed roles; filter active=true for open roles). NOT REFRESHED since ~2026-07-28 — treat as stale. Full overwrite each run (ledger is cumulative so a rebuild is correct).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.job_id IS 'Unique job posting id (ledger key).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.first_seen IS 'Date this posting was first seen by the scraper.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.last_seen IS 'Date this posting was last seen still open.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.company IS 'Hiring company, normalized to the parent level (e.g. Amazon AGI -> Amazon, Microsoft Research -> Microsoft). Mistral AI is excluded from all hiring tables (scraper not producing clean data).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.category IS 'Top-level role category (e.g. engineering, sales_gtm, research, operations, other).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.sub_area IS 'Sub-category within category.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.theme IS 'Thematic tag for the role.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.vertical IS 'Industry vertical tag when applicable (e.g. health_rd, health_delivery, agriculture, education).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.social_impact IS 'True if this role is tagged as social-impact-relevant.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.active IS 'True if the posting is currently open. The ledger is cumulative (keeps closed roles) — filter active=true for "open now" questions.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.title IS 'Job title. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.department IS 'Hiring department. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.location IS 'Job location. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.url IS 'Posting URL. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.source IS 'Scraper source identifier. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.description IS 'Full job description text. Only populated for currently-listed postings.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_jobs.captured_at IS 'Date this row was loaded (table is overwritten each run).';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.hiring_weekly IS
    'Per-company weekly posting totals (open/new/removed), recomputed from weekly_trends.json each run. NOT REFRESHED since ~2026-07-28 — treat as stale.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.week IS 'The week this row summarizes.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.company IS 'Hiring company (parent-level normalized).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.baseline IS 'True if this week is a baseline/reference week rather than a normal delta week.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.total IS 'Total open postings for this company this week.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.new IS 'Postings newly opened this week.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.removed IS 'Postings closed/removed this week.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.social_impact IS 'Count of social-impact-tagged postings this week.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly.new_social_impact IS 'Count of newly-opened social-impact postings this week.';

COMMENT ON TABLE fso_market_intelligence.frontier_labs.hiring_weekly_breakdown IS
    'Per-company weekly posting composition by tag dimension (long format), recomputed from weekly_trends.json each run. NOT REFRESHED since ~2026-07-28 — treat as stale.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly_breakdown.week IS 'The week this row summarizes.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly_breakdown.company IS 'Hiring company (parent-level normalized).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly_breakdown.dimension IS 'Which tag dimension this row breaks down by: category, sub_area, vertical, or theme.';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly_breakdown.value IS 'The specific value within that dimension (e.g. "engineering" when dimension=category).';
COMMENT ON COLUMN fso_market_intelligence.frontier_labs.hiring_weekly_breakdown.count IS 'Number of postings for this company/week/dimension/value combination.';
