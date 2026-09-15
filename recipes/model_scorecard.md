# Recipe: model scorecard → stakeholder-ready read

`model_scorecard(model)` (in `server/scorecard.py`) returns the *grounded* card:
measured capability + percentile, stated preference (arena), revealed usage
(OpenRouter), cost, and deterministic gap flags. It does not write prose — that's
this recipe's job, done by an MCP client (Claude) reasoning over the card's fields.

## Steps

1. **Call `model_scorecard(model)`.** If it raises (model not found), search with
   `query_dataset('model_benchmarks', filters={...})` to find the right name/slug and
   retry — don't guess a spelling.
2. **Read `identity` + `measured` first.** State what the model is (org, params,
   open/closed, release date) and how it stacks up (`measured.percentile` against
   `measured.peer_set_size` peers). If `measured.percentile` is `null`, say so plainly
   — don't infer a capability claim from missing data.
3. **Layer in `stated` and `revealed`, explicitly marking absence.** "Not on arena" and
   "not tracked on OpenRouter" are real answers, not gaps to paper over — say when a
   signal isn't available rather than omitting it silently.
4. **Read `gaps` and `flags` as prompts to explain, not as the verdict.** E.g. a
   `"hyped"` flag means stated percentile clears measured percentile by >0.15 — the
   narrative should say *why that might be* (agentic strengths arena captures that the
   benchmark peer set doesn't, recency, marketing) rather than restating the number.
5. **Cost + self-host:** report `cost.price_blended_3to1` / `cost.cost_per_task_total`
   plainly. For `cost.self_host`, if `self_hostable` is `false`, say self-hosting isn't
   an option (closed-weight) — don't speculate a number. If a `cheapest_fit.note`
   contains the commodity-hardware caveat, lead with that rather than a $/hr figure it
   would be misleading to feature. If it's a multi-GPU plan, say so and flag the
   interconnect-overhead caveat already present in `caveats`.
6. **Close with the "so what."** One or two sentences translating the flags into a
   decision-relevant read for the foundation's work — e.g. "underadopted" + open-weight
   + cheap self-host cost → worth a pilot; "hyped" + high cost → wait for benchmarks to
   catch up before committing budget.

## Guardrails
- Never invent a number the card didn't return. If a field is `null` or `available:
  false`, say so.
- `release_date` is the time axis (no history table) — if asked "how does this compare
  to models 6 months ago," pull other `model_benchmarks` rows by `release_date` rather
  than assuming a `model_scorecard` history exists.
- Don't inline VRAM/GPU-fit math yourself — `compute_fit` already did it inside
  `cost.self_host`; just interpret the fields.
