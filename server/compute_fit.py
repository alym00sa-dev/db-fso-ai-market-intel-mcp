"""compute_fit — deterministic VRAM/GPU-fit + concurrency calculator (Level 2).

Given a model (or raw params_b), estimates the VRAM needed to host it, which
of Ornn's 5 tracked GPUs (at which shard size, 1/2/4/8) actually serve it, and
the cheapest viable configuration's $/hr. The fit is calculated here, never
eyeballed — model_scorecard's self-host sub-read calls this instead of doing
its own math (see planning/05-compute-fit-calculator.md).

Sizing follows the "Will It Fit?" approach: weights + KV cache + a fixed
per-GPU overhead must fit within a usable fraction of total VRAM. KV cache
uses the exact architecture formula (2 x num_layers x num_kv_heads x head_dim
x bytes) when num_layers/num_kv_heads/head_dim are known (from an explicit
call argument, model_benchmarks, or server/data/model_architectures.json, in
that priority order); otherwise it falls back to a grouped-query-attention
calibrated estimate and reports the uncertainty range (see kv_mb_per_token_range
in the output) rather than presenting a false-precision number.

compute_fit evaluates every (GPU, shard size) combination up to 8 GPUs per
instance (one node) — see `configurations` in the output — and ranks every
configuration that can serve at least one concurrent request by $/hr. It does
not extrapolate to multi-node (>8 GPU) deployments.
"""

from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Any

from server import retrieval
from server.reconciliation import normalize_model_name, tokenize_model_name
from server.stats import to_float

# Weight precision -> bytes/param. 4-bit formats carry per-group quantization
# scales, so int4/fp4 are ~4.5 effective bits/param, not a clean 4.
_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5625,
    "fp4": 0.5625,
}
# KV cache precision is independent of weight precision — real deployments
# keep KV at 16-bit unless FP8 KV is explicitly chosen, regardless of how the
# weights are quantized.
_KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0}

_DEFAULT_CONTEXT_LENGTH = 8192
_SHARD_SIZES = (1, 2, 4, 8)

# Fallback KV-bytes-per-token estimate (used only when no architecture is
# known at all) is a power law in params_b, calibrated at 16-bit KV on two
# real grouped-query-attention models — GQA is the modern default, unlike the
# classic multi-head-attention anchors this curve replaces:
#   Llama-3-8B  (32 layers x 8 KV heads x 128 head_dim) -> 0.131 MB/token
#   Llama-3-70B (80 layers x 8 KV heads x 128 head_dim) -> 0.328 MB/token
_KV_GQA_COEFFICIENT = 0.0546
_KV_GQA_EXPONENT = 0.42
# Old MHA-anchored curve, kept only to report the upper end of the
# uncertainty range when the estimate is used (see kv_mb_per_token_range).
_KV_MHA_COEFFICIENT = 0.1283
_KV_MHA_EXPONENT = 0.7

# Ornn tracks exactly these 5 accelerators (planning/05-compute-fit-calculator.md).
# VRAM figures are the vendor-quoted spec per card, not measured here.
# NOTE: do not change this list or these VRAM figures without separately
# verifying GB-vs-GiB handling (see planning doc "out of scope").
GPU_SPECS: dict[str, dict[str, Any]] = {
    "B200": {"vram_gb": 192.0, "class": "datacenter"},
    "H200": {"vram_gb": 141.0, "class": "datacenter"},
    "H100 SXM": {"vram_gb": 80.0, "class": "datacenter"},
    # NOTE: A100 SXM4 also ships in a 40GB variant; we assume the 80GB variant.
    "A100 SXM4": {"vram_gb": 80.0, "class": "datacenter"},
    "RTX 5090": {"vram_gb": 32.0, "class": "consumer"},
}

_CONSUMER_NOTE = (
    "runs on commodity/consumer hardware — datacenter GPU pricing not representative"
)
_MULTI_NODE_NOTE = "requires multi-node; not evaluated"

# Lower = more worth a reader's attention. `configurations` is sorted by this
# (then cost ascending) so a scan from the top sees the usable options first —
# `does_not_load` configs sort last rather than being mixed in by GPU order.
_STATUS_PRIORITY = {
    "serves_workload": 0,
    "serves_with_replicas": 1,
    "loads_but_cannot_serve": 2,
    "does_not_load": 3,
}
_WORKING_STATUSES = {"serves_workload", "serves_with_replicas"}

# fp8/fp4 aren't natively accelerated on every GPU — these are caveats, never
# an exclusion (a config is still evaluated and can still be the cheapest
# pick), but the mismatch must be visible ON that pick, not just buried in the
# generic caveats list (see _precision_note / cheapest_fit construction).
_NATIVE_PRECISION_GPUS = {
    "fp8": {"H100 SXM", "H200", "B200", "RTX 5090"},
    "fp4": {"B200", "RTX 5090"},
}

# Local architecture overrides, keyed by model_benchmarks.slug (matched via a
# normalized/canonical comparison, not raw string equality — slug formatting
# has been observed to drift, e.g. "llama-3.3-70b-instruct" vs the real
# "llama-3-3-instruct-70b", which silently fell back to the rougher estimate
# until this was caught against live data). Populate from each model's
# Hugging Face config.json as they come up — do not invent values.
# TODO: only llama-3-3-instruct-70b is seeded; every other model still falls
# back to the GQA estimate until its real config.json values are added here.
_ARCH_OVERRIDE_PATH = Path(__file__).parent / "data" / "model_architectures.json"

_BASE_MODEL_COLUMNS = (
    "name",
    "slug",
    "params_total_b",
    "params_active_b",
    "context_window_tokens",
    "is_open_weights",
)
_ARCH_COLUMNS = ("num_layers", "num_kv_heads", "head_dim", "num_attention_heads")
# Rows tied on name-match quality that share all three of these are the same
# weights under a different mode (reasoning/non-reasoning, effort level, etc.)
# — safe to auto-pick between them rather than forcing the caller to disambiguate.
_VARIANT_TIE_COLUMNS = ("params_total_b", "params_active_b", "context_window_tokens")


def _load_arch_overrides() -> dict[str, dict[str, Any]]:
    try:
        with open(_ARCH_OVERRIDE_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    return {normalize_model_name(k): v for k, v in raw.items()}


_ARCH_OVERRIDES = _load_arch_overrides()


class ComputeFitError(ValueError):
    """Raised for bad/ambiguous inputs — fail closed rather than guess."""


def _normalize_gpu_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _available_columns(dataset: str) -> set[str]:
    return {c["name"] for c in retrieval.describe_dataset(dataset)["columns"]}


def _row_names(row: dict[str, Any]) -> list[str]:
    return [row[c] for c in ("name", "slug") if row.get(c)]


def _lookup_model_row(model: str) -> tuple[dict[str, Any], str | None, list[str] | None]:
    """Resolve `model` to a model_benchmarks row.

    Returns (row, matched_from, other_candidates). The latter two are None
    when resolved via exact name/slug equality; populated when resolved via
    fuzzy token-subset matching, so the caller can report what was actually
    matched (and what else was close).

    Tries exact equality on name, then slug (fast path, one round trip each).
    Failing that, pulls all rows and tokenizes both the query and every row's
    name/slug (see reconciliation.tokenize_model_name — lowercase, decimal
    versions like "3.3" kept as one token, a hyphen-written version like
    "3-3" merged back into "3.3"). A row matches if every query token appears
    among its tokens; matches are ranked by fewest *extra* tokens (closest
    match), ties broken by name.

    - Exactly one match at the best rank -> auto-picked.
    - Several tied at the best rank but sharing identical
      params_total_b/params_active_b/context_window_tokens (reasoning vs.
      non-reasoning, effort-level variants of the same weights) -> auto-picked
      (the first, after the name tie-break).
    - Otherwise -> ComputeFitError listing up to 5 candidates, best match
      first. Never silently picks among genuinely different models — this is
      the fix for a real bug where "llama 3.3 70b" surfaced only Llama 3.1
      rows (token-level "3.3"/"3.1" collision in an earlier, naive matcher).
    """
    available = _available_columns("model_benchmarks")
    columns = (
        [c for c in _BASE_MODEL_COLUMNS if c in available]
        + [c for c in _ARCH_COLUMNS if c in available]
    )

    for column in ("name", "slug"):
        if column not in available:
            continue
        result = retrieval.query_dataset(
            "model_benchmarks", columns=columns, filters={column: model}, limit=1
        )
        if result["rows"]:
            return result["rows"][0], None, None

    all_rows = retrieval.query_dataset("model_benchmarks", columns=columns, limit=retrieval.MAX_LIMIT)[
        "rows"
    ]
    query_tokens = set(tokenize_model_name(model))

    scored: list[tuple[int, str, dict[str, Any]]] = []
    if query_tokens:
        for row in all_rows:
            name = row.get("name") or row.get("slug") or ""
            row_tokens = set(tokenize_model_name(name))
            if query_tokens.issubset(row_tokens):
                scored.append((len(row_tokens - query_tokens), name, row))

    if not scored:
        all_names = [n for r in all_rows for n in _row_names(r)]
        suggestions = difflib.get_close_matches(model, all_names, n=5, cutoff=0.5)
        if suggestions:
            raise ComputeFitError(
                f"Model '{model}' not found in model_benchmarks. Did you mean: {suggestions}?"
            )
        raise ComputeFitError(
            f"Model '{model}' not found in model_benchmarks by name, slug, or token match, "
            "and no close matches were found. Call query_dataset('model_benchmarks', ...) to "
            "find the right identifier."
        )

    scored.sort(key=lambda t: (t[0], t[1]))
    best_extra = scored[0][0]
    tied = [t for t in scored if t[0] == best_extra]

    if len(tied) > 1:
        variant_keys = {tuple(t[2].get(c) for c in _VARIANT_TIE_COLUMNS) for t in tied}
        if len(variant_keys) > 1:
            candidates = [t[1] for t in scored[:5]]
            raise ComputeFitError(
                f"Ambiguous model '{model}' — matches: {candidates} (showing up to 5, best "
                "match first). Pass the exact name or slug."
            )

    chosen = tied[0]
    other_candidates = [t[1] for t in scored if t is not chosen][:4]
    return chosen[2], model, other_candidates


def _lookup_price(prices: dict[str, dict[str, Any]], gpu_name: str) -> dict[str, Any] | None:
    for live_name, info in prices.items():
        if _normalize_gpu_key(live_name) == _normalize_gpu_key(gpu_name):
            return info
    return None


def _latest_gpu_prices() -> dict[str, dict[str, Any]]:
    """{gpu_name (as stored live): {"date": ..., "price_usd_hr": float}}, latest date only.

    Orders by date descending so the first row seen per GPU is the newest —
    safe even once price history exceeds a single query's row limit (unlike
    pulling unordered rows and taking max(date) in Python, which silently
    picks a stale price if the newest rows don't make it into the page).
    """
    data = retrieval.query_dataset(
        "ornn_gpu_prices",
        columns=["date", "gpu_name", "index_value_usd_hr"],
        order_by="date",
        descending=True,
        limit=retrieval.MAX_LIMIT,
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in data["rows"]:
        name = row.get("gpu_name")
        date = row.get("date")
        price = to_float(row.get("index_value_usd_hr"))
        if name is None or date is None or price is None:
            continue
        if name not in latest:  # first occurrence per name is the newest (date-desc order)
            latest[name] = {"date": date, "price_usd_hr": price}
    return latest


def _resolve_architecture(
    model_row: dict[str, Any] | None,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    num_attention_heads: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """Returns (architecture_or_None, kv_source) — priority: explicit args,
    then model_benchmarks columns, then the local override file, then None
    (caller falls back to the estimate)."""
    if num_layers is not None and num_kv_heads is not None and head_dim is not None:
        return (
            {
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_attention_heads": num_attention_heads,
            },
            "user_supplied",
        )

    if model_row is not None:
        if all(model_row.get(k) is not None for k in ("num_layers", "num_kv_heads", "head_dim")):
            return (
                {
                    "num_layers": int(model_row["num_layers"]),
                    "num_kv_heads": int(model_row["num_kv_heads"]),
                    "head_dim": int(model_row["head_dim"]),
                    "num_attention_heads": (
                        int(model_row["num_attention_heads"])
                        if model_row.get("num_attention_heads") is not None
                        else None
                    ),
                },
                "model_benchmarks",
            )

        slug = model_row.get("slug")
        override = _ARCH_OVERRIDES.get(normalize_model_name(slug)) if slug else None
        if override:
            return (
                {
                    "num_layers": override["num_layers"],
                    "num_kv_heads": override["num_kv_heads"],
                    "head_dim": override["head_dim"],
                    "num_attention_heads": override.get("num_attention_heads"),
                },
                "local_override",
            )

    return None, "estimated"


def _kv_sizing(
    arch: dict[str, Any] | None, kv_source: str, params_b: float, kv_bytes: float
) -> tuple[float, list[float] | None]:
    """Returns (kv_bytes_per_token, kv_mb_per_token_range_or_None)."""
    if arch is not None:
        bytes_per_token = 2 * arch["num_layers"] * arch["num_kv_heads"] * arch["head_dim"] * kv_bytes
        return bytes_per_token, None

    scale = kv_bytes / 2.0  # anchors are calibrated at 16-bit
    gqa_mb = _KV_GQA_COEFFICIENT * (params_b**_KV_GQA_EXPONENT) * scale
    mha_mb = _KV_MHA_COEFFICIENT * (params_b**_KV_MHA_EXPONENT) * scale
    return gqa_mb * 1e6, [round(gqa_mb, 4), round(mha_mb, 4)]


def _shard_sizes_for(spec: dict[str, Any]) -> tuple[int, ...]:
    """Consumer cards (e.g. RTX 5090) lack the fast interconnect (NVLink)
    datacenter cards have, so sharding a model across several of them isn't a
    realistic deployment — only evaluate n=1 for the consumer class."""
    if spec["class"] == "consumer":
        return (1,)
    return _SHARD_SIZES


def _precision_note(precision: str, gpu_name: str) -> str | None:
    """None if `precision` is native on `gpu_name` (or isn't one we track
    native-support nuances for); otherwise the caveat text — attached directly
    to the configuration (and propagated into cheapest_fit), not left to the
    generic top-level caveats list where it's easy to miss on the actual pick."""
    native_gpus = _NATIVE_PRECISION_GPUS.get(precision)
    if native_gpus is None or gpu_name in native_gpus:
        return None
    return (
        f"{precision} isn't natively accelerated on {gpu_name} — weight-only kernels may "
        "still run it, but throughput/efficiency may suffer."
    )


def _ruled_out_reason(cfg: dict[str, Any], precision: str) -> str:
    """One short phrase for why a configuration isn't in the trimmed
    `configurations` list (see include_all_configurations=False)."""
    if cfg["status"] == "does_not_load":
        return "weights don't fit"
    if cfg["status"] == "loads_but_cannot_serve":
        return "loads but no room for conversations"
    if not cfg["precision_native"]:
        return f"{precision} not native (cheaper option shown separately)"
    return "cost or capacity below the top options shown"


def compute_fit(
    model: str | None = None,
    params_b: float | None = None,
    precision: str = "fp16",
    kv_precision: str = "bf16",
    context_length: int | None = None,
    batch_size: int = 1,
    overhead_gb_per_gpu: float = 3.0,
    usable_fraction: float = 0.9,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    num_attention_heads: int | None = None,
    overhead_factor: float | None = None,
    include_all_configurations: bool = False,
) -> dict[str, Any]:
    """Compute VRAM + concurrency fit for a model across Ornn's tracked GPUs,
    and recommend the cheapest configuration that natively supports the
    requested precision.

    Pass either `model` (looked up in model_benchmarks by name/slug, with a
    forgiving token-based fallback — see _lookup_model_row; when the match
    wasn't an exact name/slug hit, the output includes `matched_from` and
    `other_candidates` so the caller can say what was matched) or `params_b`
    directly. `context_length` always defaults to 8192 regardless of the
    model's actual context window (a self-host read sized to a 128K window by
    default would badly overstate typical request cost) — pass it explicitly
    to size for a different request length, or see `max_context_stress` for
    what a single full-context-window request would need.

    `cheapest_fit` is the cheapest configuration that both serves the
    requested `batch_size` (or the closest it can, via replicas) AND natively
    supports `precision` — a GPU that technically works via weight-only
    kernels but doesn't natively accelerate the chosen precision is never the
    headline unless nothing native works at all. If a cheaper non-native
    configuration exists, it's surfaced separately as
    `cheaper_non_native_option` (with `savings_usd_hr`) rather than silently
    omitted or presented as the default.

    By default (`include_all_configurations=False`), `configurations` holds
    only the top 5 working configurations (see `ruled_out` for a compact
    summary of what else was evaluated and why it isn't shown); pass
    `include_all_configurations=True` for the full evaluated set.

    `precision` sizes the weights; `kv_precision` sizes the KV cache
    independently (KV usually stays 16-bit even when weights are quantized).
    `num_layers`/`num_kv_heads`/`head_dim`/`num_attention_heads` let a caller
    size the KV cache exactly; otherwise architecture is looked up from
    model_benchmarks or server/data/model_architectures.json, falling back to
    a grouped-query-attention-calibrated estimate (reported as a range, see
    `kv_mb_per_token_range`) when nothing is known.

    For each of Ornn's 5 GPUs, evaluates shard sizes 1/2/4/8 (one node, never
    extrapolated further) and returns every configuration's status and
    concurrency/cost in `configurations`; `fits_on`/`cheapest_fit` surface the
    single-GPU view and the overall cheapest configuration that can serve the
    requested `batch_size`, respectively.

    `overhead_factor` is deprecated and ignored — pass `overhead_gb_per_gpu`
    instead (a flat per-GPU memory overhead, not a multiplier on weights).
    """
    if model is None and params_b is None:
        raise ComputeFitError("Pass either 'model' or 'params_b'.")
    if precision not in _BYTES_PER_PARAM:
        raise ComputeFitError(f"Unknown precision '{precision}'. Must be one of {sorted(_BYTES_PER_PARAM)}.")
    if kv_precision not in _KV_BYTES:
        raise ComputeFitError(
            f"Unknown kv_precision '{kv_precision}'. Must be one of {sorted(_KV_BYTES)}."
        )
    if batch_size < 1:
        raise ComputeFitError("batch_size must be >= 1.")
    if context_length is not None and context_length < 1:
        raise ComputeFitError("context_length must be >= 1.")
    if not (0 < usable_fraction <= 1):
        raise ComputeFitError("usable_fraction must be in (0, 1].")

    is_open_weights: bool | None = None
    resolved_model_name: str | None = None
    model_row: dict[str, Any] | None = None
    model_context_window: float | None = None
    matched_from: str | None = None
    other_candidates: list[str] | None = None
    params_source: str | None = "user_supplied" if params_b is not None else None

    if model is not None:
        model_row, matched_from, other_candidates = _lookup_model_row(model)
        resolved_model_name = model_row.get("name") or model_row.get("slug")
        is_open_weights = model_row.get("is_open_weights")
        if params_b is None:
            slug = model_row.get("slug")
            override = _ARCH_OVERRIDES.get(normalize_model_name(slug)) if slug else None
            override_params_b = to_float(override.get("params_total_b")) if override else None
            if override_params_b is not None:
                params_b = override_params_b
                params_source = "local_override"
            else:
                params_b = to_float(model_row.get("params_total_b"))
                params_source = "model_benchmarks"
            if params_b is None:
                raise ComputeFitError(
                    f"model_benchmarks has no params_total_b for '{resolved_model_name}'. "
                    "Pass params_b explicitly."
                )
        model_context_window = to_float(model_row.get("context_window_tokens"))

    if params_b is None or params_b <= 0:
        raise ComputeFitError("params_b must be > 0.")

    if is_open_weights is False:
        closed_result: dict[str, Any] = {
            "model": resolved_model_name or model,
            "self_hostable": False,
            "reason": "Model is closed-weight (API-only) — self-host isn't an option.",
        }
        if matched_from is not None:
            closed_result["matched_from"] = matched_from
            closed_result["other_candidates"] = other_candidates
        return closed_result

    user_supplied_context = context_length is not None
    context_length = context_length or _DEFAULT_CONTEXT_LENGTH

    bytes_per_param = _BYTES_PER_PARAM[precision]
    kv_bytes = _KV_BYTES[kv_precision]

    arch, kv_source = _resolve_architecture(model_row, num_layers, num_kv_heads, head_dim, num_attention_heads)
    kv_bytes_per_token, kv_mb_per_token_range = _kv_sizing(arch, kv_source, params_b, kv_bytes)
    resolved_num_attention_heads = arch["num_attention_heads"] if arch else None

    weights_gb = params_b * bytes_per_param
    kv_per_request_gb = kv_bytes_per_token * context_length / 1e9
    kv_gb = kv_per_request_gb * batch_size
    total_gb = weights_gb + kv_gb + overhead_gb_per_gpu  # single-GPU (n=1) view
    required_gb_with_headroom = total_gb / usable_fraction

    prices = _latest_gpu_prices()

    configurations: list[dict[str, Any]] = []
    for gpu_name, spec in GPU_SPECS.items():
        price_info = _lookup_price(prices, gpu_name)
        price_per_gpu = price_info["price_usd_hr"] if price_info else None
        price_as_of = price_info["date"] if price_info else None

        for n in _shard_sizes_for(spec):
            if resolved_num_attention_heads is not None and resolved_num_attention_heads % n != 0:
                continue

            usable_gb = usable_fraction * n * spec["vram_gb"]
            fixed_gb = weights_gb + n * overhead_gb_per_gpu
            kv_pool_gb = usable_gb - fixed_gb
            max_concurrent = math.floor(kv_pool_gb / kv_per_request_gb) if kv_pool_gb > 0 else 0
            token_budget = math.floor(kv_pool_gb * 1e9 / kv_bytes_per_token) if kv_pool_gb > 0 else 0

            if kv_pool_gb <= 0:
                status = "does_not_load"
            elif max_concurrent == 0:
                status = "loads_but_cannot_serve"
            elif max_concurrent < batch_size:
                # NOT a failure — replicas (below) cover the gap. This status
                # means "needs horizontal scaling to hit batch_size", not
                # "broken"; a config here that becomes cheapest_fit still gets
                # a correctly-sized replicas/total_gpus/cost_usd_hr.
                status = "serves_with_replicas"
            else:
                status = "serves_workload"

            replicas = math.ceil(batch_size / max_concurrent) if max_concurrent >= 1 else None
            total_gpus = n * replicas if replicas is not None else None
            cost_usd_hr = (
                total_gpus * price_per_gpu if total_gpus is not None and price_per_gpu is not None else None
            )
            cost_per_concurrent = (
                cost_usd_hr / (max_concurrent * replicas)
                if cost_usd_hr is not None and max_concurrent >= 1 and replicas
                else None
            )
            weights_share = weights_gb / usable_gb if usable_gb > 0 else None

            notes: list[str] = []
            # Gated on status != does_not_load: a config that can't even load
            # weights shouldn't also claim "little room for concurrency" —
            # that note only makes sense once weights actually fit.
            if weights_share is not None and weights_share > 0.67 and status != "does_not_load":
                notes.append(
                    "weights exceed ~2/3 of usable memory; little room for KV cache / concurrency."
                )
            precision_note = _precision_note(precision, gpu_name)
            if precision_note:
                notes.append(precision_note)

            configurations.append(
                {
                    "gpu_name": gpu_name,
                    "class": spec["class"],
                    "n_gpus_per_instance": n,
                    "replicas": replicas,
                    "total_gpus": total_gpus,
                    "status": status,
                    "precision_native": precision_note is None,
                    "usable_gb": round(usable_gb, 2),
                    "kv_pool_gb": round(kv_pool_gb, 2),
                    "token_budget": token_budget,
                    "max_concurrent_requests": max_concurrent,
                    "weights_share_of_usable": round(weights_share, 3) if weights_share is not None else None,
                    "price_usd_hr_per_gpu": price_per_gpu,
                    "cost_usd_hr": round(cost_usd_hr, 4) if cost_usd_hr is not None else None,
                    "cost_usd_hr_per_concurrent_request": (
                        round(cost_per_concurrent, 4) if cost_per_concurrent is not None else None
                    ),
                    "price_as_of": price_as_of,
                    "notes": notes,
                }
            )

    # Ranking tie-break context: flag n>1 configs when that GPU also serves at n=1.
    serving_shards_by_gpu: dict[str, set[int]] = {}
    for cfg in configurations:
        if cfg["max_concurrent_requests"] >= 1:
            serving_shards_by_gpu.setdefault(cfg["gpu_name"], set()).add(cfg["n_gpus_per_instance"])
    for cfg in configurations:
        if (
            cfg["n_gpus_per_instance"] > 1
            and cfg["max_concurrent_requests"] >= 1
            and 1 in serving_shards_by_gpu.get(cfg["gpu_name"], set())
        ):
            cfg["notes"].append(
                f"{cfg['gpu_name']} also serves this workload at n=1 — prefer independent "
                "replicas over sharding for throughput and fault isolation."
            )

    # Unified sort: works before doesn't, native-precision before not, then
    # cost ascending, then total capacity descending (at equal cost, the setup
    # that serves more concurrent users wins), then fewer GPUs per instance.
    # Native preference lives IN this key (not a separate post-hoc rule) so
    # configurations[0] and cheapest_fit can never disagree, by construction.
    def _sort_key(c: dict[str, Any]) -> tuple[Any, ...]:
        return (
            c["status"] not in _WORKING_STATUSES,
            not c["precision_native"],
            c["cost_usd_hr"] is None,
            c["cost_usd_hr"] or 0.0,
            -(c["max_concurrent_requests"] * (c["replicas"] or 1)),
            c["n_gpus_per_instance"],
        )

    configurations.sort(key=_sort_key)
    working_configs = [c for c in configurations if c["status"] in _WORKING_STATUSES]

    fits_on = [
        {
            "gpu_name": c["gpu_name"],
            "class": c["class"],
            "vram_gb": GPU_SPECS[c["gpu_name"]]["vram_gb"],
            "price_usd_hr": c["price_usd_hr_per_gpu"],
            "price_as_of": c["price_as_of"],
        }
        for c in configurations
        if c["n_gpus_per_instance"] == 1 and c["status"] == "serves_workload"
    ]
    fits_on.sort(key=lambda g: (g["price_usd_hr"] is None, g["price_usd_hr"] or 0.0))

    cheaper_non_native_option: dict[str, Any] | None = None

    if working_configs:
        top = working_configs[0]  # already native-preferring by _sort_key
        top_precision_note = _precision_note(precision, top["gpu_name"])
        if top["n_gpus_per_instance"] == 1 and top["replicas"] == 1:
            matching = next(f for f in fits_on if f["gpu_name"] == top["gpu_name"])
            if matching["class"] == "consumer":
                cheapest_fit: dict[str, Any] = {"note": _CONSUMER_NOTE, "matching_gpu": matching}
            else:
                cheapest_fit = dict(matching)
                cheapest_fit["note"] = None
            if top_precision_note:
                # Only reached when NO native option works at all (otherwise
                # _sort_key already put a native config first) — merge into
                # the same field a reader sees first either way.
                cheapest_fit["note"] = (
                    f"{cheapest_fit['note']} — {top_precision_note}"
                    if cheapest_fit.get("note")
                    else top_precision_note
                )
            if cheapest_fit.get("note") is None:
                del cheapest_fit["note"]
        else:
            note = (
                f"needs {top['n_gpus_per_instance']}x {top['gpu_name']} per instance, "
                f"{top['replicas']} replica(s) for batch_size={batch_size}"
            )
            if top_precision_note:
                note = f"{note} — {top_precision_note}"
            cheapest_fit = {
                "n_gpus": top["n_gpus_per_instance"],
                "gpu_name": top["gpu_name"],
                "replicas": top["replicas"],
                "cost_usd_hr": top["cost_usd_hr"],
                "note": note,
            }

        # Cheapest working config by cost alone (ignoring native preference) —
        # if that's non-native and actually cheaper than the native-preferring
        # cheapest_fit, surface it separately rather than hide a real option.
        cheapest_overall = min(
            working_configs,
            key=lambda c: (
                c["cost_usd_hr"] is None,
                c["cost_usd_hr"] or 0.0,
                -(c["max_concurrent_requests"] * (c["replicas"] or 1)),
                c["n_gpus_per_instance"],
            ),
        )
        if (
            not cheapest_overall["precision_native"]
            and cheapest_overall["cost_usd_hr"] is not None
            and top["cost_usd_hr"] is not None
            and cheapest_overall["cost_usd_hr"] < top["cost_usd_hr"]
        ):
            cheaper_non_native_option = {
                "gpu_name": cheapest_overall["gpu_name"],
                "n_gpus_per_instance": cheapest_overall["n_gpus_per_instance"],
                "replicas": cheapest_overall["replicas"],
                "total_gpus": cheapest_overall["total_gpus"],
                "cost_usd_hr": cheapest_overall["cost_usd_hr"],
                "savings_usd_hr": round(top["cost_usd_hr"] - cheapest_overall["cost_usd_hr"], 4),
                "note": (
                    f"{precision} isn't natively accelerated on {cheapest_overall['gpu_name']}; "
                    "may run via weight-only kernels with lower throughput. Test before "
                    "relying on it."
                ),
            }
    else:
        cheapest_fit = {"note": _MULTI_NODE_NOTE}

    caveats: list[str] = []
    if precision in ("fp16", "bf16"):
        caveats.append(
            f"Weights sized at {precision} ({bytes_per_param} bytes/param); quantization "
            "(fp8/int8/int4/fp4) materially lowers this."
        )
    if precision == "fp8":
        caveats.append(
            "fp8 is natively accelerated on H100 SXM, H200, B200, and RTX 5090; A100 SXM4 "
            "isn't native (weight-only fp8 kernels may still run it)."
        )
    if precision == "fp4":
        caveats.append("fp4 is natively accelerated only on Blackwell-class cards (B200, RTX 5090) here.")
    if kv_source == "estimated":
        caveats.append(
            "KV cache is estimated (no architecture available): assumes grouped-query "
            "attention — could be several times low if the model uses classic "
            "multi-head attention. See kv_mb_per_token_range for the spread."
        )
    if any(c["gpu_name"] == "A100 SXM4" for c in working_configs):
        caveats.append("A100 SXM4 assumed as the 80GB variant; a 40GB variant also exists in the market.")
    if any(c["n_gpus_per_instance"] > 1 for c in working_configs):
        caveats.append(
            "Multi-GPU throughput doesn't scale linearly with GPU count; interconnect "
            "(NVLink within a node) matters."
        )
    if overhead_factor is not None:
        caveats.append("overhead_factor is deprecated and ignored — use overhead_gb_per_gpu instead.")
    if user_supplied_context and model_context_window and context_length > model_context_window:
        caveats.append(
            f"context_length ({context_length}) exceeds this model's context window "
            f"({int(model_context_window)}) — using your value anyway."
        )
    caveats.append("Ornn tracks only these 5 accelerators — a spot index, not the full hardware market.")

    if include_all_configurations:
        visible_configurations = configurations
        ruled_out = None
    else:
        _TOP_N = 5
        top_working = working_configs[:_TOP_N]
        kept_keys = {(c["gpu_name"], c["n_gpus_per_instance"]) for c in top_working}
        visible_configurations = top_working
        ruled_out = [
            {
                "gpu_name": c["gpu_name"],
                "n_gpus_per_instance": c["n_gpus_per_instance"],
                "status": c["status"],
                "reason": _ruled_out_reason(c, precision),
            }
            for c in configurations
            if (c["gpu_name"], c["n_gpus_per_instance"]) not in kept_keys
        ]

    result: dict[str, Any] = {
        "model": resolved_model_name or model,
        "self_hostable": True,
        "params_b": params_b,
        "params_source": params_source,
        "precision": precision,
        "assumptions": {
            "bytes_per_param": bytes_per_param,
            "kv_precision": kv_precision,
            "kv_bytes_per_token": round(kv_bytes_per_token, 2),
            "kv_source": kv_source,
            "overhead_gb_per_gpu": overhead_gb_per_gpu,
            "usable_fraction": usable_fraction,
            "context_length": context_length,
            "batch_size": batch_size,
            "overhead_factor": overhead_factor,
        },
        "weights_gb": round(weights_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "total_gb": round(total_gb, 2),
        "required_gb_with_headroom": round(required_gb_with_headroom, 2),
        "fits_on": fits_on,
        "cheapest_fit": cheapest_fit,
        "configurations": visible_configurations,
        "caveats": caveats,
    }

    if matched_from is not None:
        result["matched_from"] = matched_from
        result["other_candidates"] = other_candidates
    if ruled_out is not None:
        result["ruled_out"] = ruled_out
    if cheaper_non_native_option is not None:
        result["cheaper_non_native_option"] = cheaper_non_native_option
    if kv_mb_per_token_range is not None:
        result["kv_mb_per_token_range"] = kv_mb_per_token_range

    if model_context_window:
        full_window_kv_per_request = kv_bytes_per_token * model_context_window / 1e9
        visible_keys = {(c["gpu_name"], c["n_gpus_per_instance"]) for c in visible_configurations}
        result["max_context_stress"] = {
            "full_context_window_tokens": int(model_context_window),
            "kv_per_request_at_full_window_gb": round(full_window_kv_per_request, 2),
            "configurations": [
                {
                    "gpu_name": c["gpu_name"],
                    "n_gpus_per_instance": c["n_gpus_per_instance"],
                    "fits_single_full_window_request": c["kv_pool_gb"] >= full_window_kv_per_request,
                }
                for c in configurations
                if (c["gpu_name"], c["n_gpus_per_instance"]) in visible_keys
            ],
        }

    return result
