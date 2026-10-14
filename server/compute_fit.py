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

import json
import math
import re
from pathlib import Path
from typing import Any

from server import retrieval
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

# Local architecture overrides, keyed by model_benchmarks.slug, for models
# where num_layers/num_kv_heads/head_dim aren't (yet) columns on
# model_benchmarks itself. Populate from each model's Hugging Face
# config.json as they come up — do not invent values.
# TODO: only llama-3.3-70b-instruct is seeded; every other model still falls
# back to the GQA estimate until its real config.json values are added here.
_ARCH_OVERRIDE_PATH = Path(__file__).parent / "data" / "model_architectures.json"

_BASE_MODEL_COLUMNS = ("name", "slug", "params_total_b", "context_window_tokens", "is_open_weights")
_ARCH_COLUMNS = ("num_layers", "num_kv_heads", "head_dim", "num_attention_heads")


def _load_arch_overrides() -> dict[str, dict[str, Any]]:
    try:
        with open(_ARCH_OVERRIDE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_ARCH_OVERRIDES = _load_arch_overrides()


class ComputeFitError(ValueError):
    """Raised for bad/ambiguous inputs — fail closed rather than guess."""


def _normalize_gpu_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _available_columns(dataset: str) -> set[str]:
    return {c["name"] for c in retrieval.describe_dataset(dataset)["columns"]}


def _lookup_model_row(model: str) -> dict[str, Any]:
    """Resolve `model` to a model_benchmarks row.

    Tries exact equality on name, then slug (fast path, one round trip each).
    Failing that, pulls all rows and matches on a normalized key (lowercase,
    [a-z0-9] only): first a unique exact normalized match, then a unique row
    whose normalized name/slug contains the normalized query. Zero or
    multiple matches at the end raise ComputeFitError listing up to 5
    candidates — never auto-picks among ambiguous matches.
    """
    available = _available_columns("model_benchmarks")
    columns = [c for c in _BASE_MODEL_COLUMNS if c in available] + [
        c for c in _ARCH_COLUMNS if c in available
    ]

    for column in ("name", "slug"):
        if column not in available:
            continue
        result = retrieval.query_dataset(
            "model_benchmarks", columns=columns, filters={column: model}, limit=1
        )
        if result["rows"]:
            return result["rows"][0]

    all_rows = retrieval.query_dataset("model_benchmarks", columns=columns, limit=retrieval.MAX_LIMIT)[
        "rows"
    ]
    query_norm = _normalize_name(model)

    def normalized_fields(row: dict[str, Any]) -> list[str]:
        return [_normalize_name(row[c]) for c in ("name", "slug") if row.get(c)]

    exact = [r for r in all_rows if query_norm in normalized_fields(r)]
    if len(exact) == 1:
        return exact[0]

    contains = [r for r in all_rows if any(query_norm in nf for nf in normalized_fields(r))]
    if len(contains) == 1:
        return contains[0]

    pool = exact if len(exact) > 1 else contains
    if pool:
        candidates = [r.get("name") or r.get("slug") for r in pool[:5]]
        raise ComputeFitError(
            f"Ambiguous model '{model}' — matches: {candidates} (showing up to 5). "
            "Pass the exact name or slug."
        )
    raise ComputeFitError(
        f"Model '{model}' not found in model_benchmarks by name, slug, or normalized "
        "match. Call query_dataset('model_benchmarks', ...) to find the right identifier."
    )


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
        override = _ARCH_OVERRIDES.get(slug) if slug else None
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
) -> dict[str, Any]:
    """Compute VRAM + concurrency fit for a model across Ornn's tracked GPUs.

    Pass either `model` (looked up in model_benchmarks by name/slug, with a
    forgiving normalized-name fallback — see _lookup_model_row) or `params_b`
    directly. `context_length` always defaults to 8192 regardless of the
    model's actual context window (a self-host read sized to a 128K window by
    default would badly overstate typical request cost) — pass it explicitly
    to size for a different request length, or see `max_context_stress` for
    what a single full-context-window request would need.

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

    if model is not None:
        model_row = _lookup_model_row(model)
        resolved_model_name = model_row.get("name") or model_row.get("slug")
        is_open_weights = model_row.get("is_open_weights")
        if params_b is None:
            params_b = to_float(model_row.get("params_total_b"))
            if params_b is None:
                raise ComputeFitError(
                    f"model_benchmarks has no params_total_b for '{resolved_model_name}'. "
                    "Pass params_b explicitly."
                )
        model_context_window = to_float(model_row.get("context_window_tokens"))

    if params_b is None or params_b <= 0:
        raise ComputeFitError("params_b must be > 0.")

    if is_open_weights is False:
        return {
            "model": resolved_model_name or model,
            "self_hostable": False,
            "reason": "Model is closed-weight (API-only) — self-host isn't an option.",
        }

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

        for n in _SHARD_SIZES:
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
                status = "serves_below_target"
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
            if weights_share is not None and weights_share > 0.67:
                notes.append(
                    "weights exceed ~2/3 of usable memory; little room for KV cache / concurrency."
                )

            configurations.append(
                {
                    "gpu_name": gpu_name,
                    "class": spec["class"],
                    "n_gpus_per_instance": n,
                    "replicas": replicas,
                    "total_gpus": total_gpus,
                    "status": status,
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

    serving_configs = [c for c in configurations if c["max_concurrent_requests"] >= 1]
    serving_configs.sort(
        key=lambda c: (c["cost_usd_hr"] is None, c["cost_usd_hr"] or 0.0, c["n_gpus_per_instance"])
    )

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

    if serving_configs:
        top = serving_configs[0]
        if top["n_gpus_per_instance"] == 1 and top["replicas"] == 1:
            matching = next(f for f in fits_on if f["gpu_name"] == top["gpu_name"])
            if matching["class"] == "consumer":
                cheapest_fit: dict[str, Any] = {"note": _CONSUMER_NOTE, "matching_gpu": matching}
            else:
                cheapest_fit = matching
        else:
            cheapest_fit = {
                "n_gpus": top["n_gpus_per_instance"],
                "gpu_name": top["gpu_name"],
                "replicas": top["replicas"],
                "cost_usd_hr": top["cost_usd_hr"],
                "note": (
                    f"needs {top['n_gpus_per_instance']}x {top['gpu_name']} per instance, "
                    f"{top['replicas']} replica(s) for batch_size={batch_size}"
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
    if any(c["gpu_name"] == "A100 SXM4" for c in serving_configs):
        caveats.append("A100 SXM4 assumed as the 80GB variant; a 40GB variant also exists in the market.")
    if any(c["n_gpus_per_instance"] > 1 for c in serving_configs):
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

    result: dict[str, Any] = {
        "model": resolved_model_name or model,
        "self_hostable": True,
        "params_b": params_b,
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
        "configurations": configurations,
        "caveats": caveats,
    }

    if kv_mb_per_token_range is not None:
        result["kv_mb_per_token_range"] = kv_mb_per_token_range

    if model_context_window:
        full_window_kv_per_request = kv_bytes_per_token * model_context_window / 1e9
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
            ],
        }

    return result
