"""compute_fit — deterministic VRAM/GPU-fit calculator (Level 2).

Given a model (or raw params_b), estimates the VRAM needed to host it and
which of the 5 GPUs Ornn tracks actually fit, then quotes the cheapest
fitting $/hr. The fit is calculated here, never eyeballed — model_scorecard's
self-host sub-read calls this instead of doing its own math (see
planning/05-compute-fit-calculator.md, the spec this module implements).
"""

from __future__ import annotations

import math
from typing import Any

from server import retrieval
from server.stats import to_float

_BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
_DEFAULT_CONTEXT_LENGTH = 8192
_SAFETY_HEADROOM = 1.1

# KV-cache-per-token approximation (see _kv_cache_gb below). num_layers x
# d_model (what KV-cache-per-token actually scales with) grows sub-linearly
# with total params — e.g. Llama-2 7B (32 layers x 4096) -> 70B (80 x 8192) is
# a 5x jump in layers*d_model against a 10x jump in params. A plain
# linear-in-params estimate overstates large models, so we use a power law
# fit to two real anchor points (both standard MHA, fp16 KV):
#   7B  ~0.5 MB/token   (Llama-2-7B: 2 x 32 layers x 4096 x 2 bytes)
#   70B ~2.5 MB/token   (Llama-2-70B-shaped MHA-equivalent: 2 x 80 x 8192 x 2 bytes)
# -> kv_mb_per_token ~= _KV_COEFFICIENT * params_b ** _KV_EXPONENT
_KV_COEFFICIENT = 0.1283
_KV_EXPONENT = 0.7

# Ornn tracks exactly these 5 accelerators (planning/05-compute-fit-calculator.md).
# VRAM figures are the vendor-quoted spec per card, not measured here.
GPU_SPECS: dict[str, dict[str, Any]] = {
    "B200": {"vram_gb": 192.0, "class": "datacenter"},
    "H200": {"vram_gb": 141.0, "class": "datacenter"},
    "H100 SXM": {"vram_gb": 80.0, "class": "datacenter"},
    # NOTE: A100 SXM4 also ships in a 40GB variant; we assume the 80GB variant
    # per the spec doc. Flagged explicitly in the output caveats.
    "A100 SXM4": {"vram_gb": 80.0, "class": "datacenter"},
    "RTX 5090": {"vram_gb": 32.0, "class": "consumer"},
}

_CONSUMER_NOTE = (
    "runs on commodity/consumer hardware — datacenter GPU pricing not representative"
)


class ComputeFitError(ValueError):
    """Raised for bad/ambiguous inputs — fail closed rather than guess."""


def _normalize_gpu_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


_GPU_SPECS_BY_NORMALIZED_NAME = {_normalize_gpu_key(k): k for k in GPU_SPECS}


def _lookup_model_row(model: str) -> dict[str, Any] | None:
    for column in ("name", "slug"):
        result = retrieval.query_dataset(
            "model_benchmarks",
            columns=["name", "slug", "params_total_b", "context_window_tokens", "is_open_weights"],
            filters={column: model},
            limit=1,
        )
        if result["rows"]:
            return result["rows"][0]
    return None


def _latest_gpu_prices() -> dict[str, dict[str, Any]]:
    """{gpu_name (as stored live): {"date": ..., "price_usd_hr": float}} at the latest date."""
    data = retrieval.query_dataset(
        "ornn_gpu_prices",
        columns=["date", "gpu_name", "index_value_usd_hr"],
        limit=retrieval.MAX_LIMIT,
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in data["rows"]:
        name = row.get("gpu_name")
        date = row.get("date")
        price = to_float(row.get("index_value_usd_hr"))
        if name is None or date is None or price is None:
            continue
        if name not in latest or date > latest[name]["date"]:
            latest[name] = {"date": date, "price_usd_hr": price}
    return latest


def _kv_cache_gb(params_b: float, bytes_per_param: float, context_length: int, batch_size: int) -> float:
    """Estimate-only, no architecture data available from model_benchmarks.

    Real KV-cache size is 2 x num_layers x kv_hidden x context x batch x bytes;
    without num_layers/kv_heads/head_dim we approximate kv-bytes-per-token as a
    power-law function of params_b, calibrated against standard multi-head-
    attention models at two real sizes (see _KV_COEFFICIENT/_KV_EXPONENT
    above). This still OVERESTIMATES for modern grouped/multi-query-attention
    models, which have much smaller KV caches per param than classic MHA.
    Treat as an upper-bound estimate, not a measurement.
    """
    kv_mb_per_token = _KV_COEFFICIENT * (params_b**_KV_EXPONENT) * (bytes_per_param / 2.0)
    kv_bytes_per_token = kv_mb_per_token * 1e6
    return (kv_bytes_per_token * context_length * batch_size) / 1e9


def compute_fit(
    model: str | None = None,
    params_b: float | None = None,
    precision: str = "fp16",
    context_length: int | None = None,
    batch_size: int = 1,
    overhead_factor: float = 1.2,
) -> dict[str, Any]:
    """Compute VRAM needed to host a model and which tracked GPU(s) fit.

    Pass either `model` (a name/slug looked up in model_benchmarks) or
    `params_b` directly. Returns weights/KV/total VRAM, the fitting GPU(s)
    from Ornn's 5-GPU set with current $/hr, and every assumption used. See
    planning/05-compute-fit-calculator.md for the full spec this implements.
    """
    if model is None and params_b is None:
        raise ComputeFitError("Pass either 'model' or 'params_b'.")
    if precision not in _BYTES_PER_PARAM:
        raise ComputeFitError(
            f"Unknown precision '{precision}'. Must be one of {sorted(_BYTES_PER_PARAM)}."
        )

    is_open_weights: bool | None = None
    resolved_model_name: str | None = None
    model_row: dict[str, Any] | None = None

    if model is not None:
        model_row = _lookup_model_row(model)
        if model_row is None:
            raise ComputeFitError(
                f"Model '{model}' not found in model_benchmarks by name or slug. "
                "Call query_dataset('model_benchmarks', ...) to find the right identifier."
            )
        resolved_model_name = model_row.get("name") or model_row.get("slug")
        is_open_weights = model_row.get("is_open_weights")
        if params_b is None:
            params_b = to_float(model_row.get("params_total_b"))
            if params_b is None:
                raise ComputeFitError(
                    f"model_benchmarks has no params_total_b for '{resolved_model_name}'. "
                    "Pass params_b explicitly."
                )
        if context_length is None:
            context_length = int(to_float(model_row.get("context_window_tokens")) or 0) or None

    if is_open_weights is False:
        return {
            "model": resolved_model_name or model,
            "self_hostable": False,
            "reason": "Model is closed-weight (API-only) — self-host isn't an option.",
        }

    context_length = context_length or _DEFAULT_CONTEXT_LENGTH
    bytes_per_param = _BYTES_PER_PARAM[precision]

    weights_gb = params_b * bytes_per_param
    model_gb = weights_gb * overhead_factor
    kv_gb = _kv_cache_gb(params_b, bytes_per_param, context_length, batch_size)
    total_gb = model_gb + kv_gb
    required_gb = total_gb * _SAFETY_HEADROOM

    prices = _latest_gpu_prices()

    fits_on = []
    for gpu_name, spec in GPU_SPECS.items():
        live_row = None
        for live_name, price_info in prices.items():
            if _normalize_gpu_key(live_name) == _normalize_gpu_key(gpu_name):
                live_row = price_info
                break
        if spec["vram_gb"] >= required_gb:
            fits_on.append(
                {
                    "gpu_name": gpu_name,
                    "class": spec["class"],
                    "vram_gb": spec["vram_gb"],
                    "price_usd_hr": live_row["price_usd_hr"] if live_row else None,
                    "price_as_of": live_row["date"] if live_row else None,
                }
            )

    caveats = [
        f"Weights sized at {precision} ({bytes_per_param} bytes/param); quantization "
        "(int8/int4) materially lowers this — this is an upper bound at the chosen precision.",
        "KV cache is an estimate: model_benchmarks has no architecture (num_layers/"
        "kv_heads/head_dim), so it's approximated from params_b calibrated against a "
        "standard MHA model — can overestimate for modern GQA/MQA architectures.",
        "A100 SXM4 assumed as the 80GB variant; a 40GB variant also exists in the market.",
        "Ornn tracks only these 5 accelerators — a spot index, not the full hardware market.",
    ]

    fits_on.sort(key=lambda g: (g["price_usd_hr"] is None, g["price_usd_hr"] or 0.0))

    if fits_on:
        cheapest = fits_on[0]
        if cheapest["class"] == "consumer":
            cheapest_fit: dict[str, Any] = {"note": _CONSUMER_NOTE, "matching_gpu": cheapest}
        else:
            cheapest_fit = cheapest
    else:
        largest_name, largest_spec = max(GPU_SPECS.items(), key=lambda kv: kv[1]["vram_gb"])
        n = math.ceil(required_gb / largest_spec["vram_gb"])
        cheapest_fit = {
            "note": f"needs {n}x {largest_name} (no single tracked GPU fits)",
            "n_gpus": n,
            "gpu_name": largest_name,
            "caveat": "multi-GPU estimate ignores inter-GPU/interconnect overhead — approximate.",
        }

    return {
        "model": resolved_model_name or model,
        # We only reach here when is_open_weights isn't explicitly False (the
        # early-return above handles that) — either it's True, or unknown
        # because the caller passed raw params_b with no model lookup.
        "self_hostable": True,
        "params_b": params_b,
        "precision": precision,
        "assumptions": {
            "bytes_per_param": bytes_per_param,
            "overhead_factor": overhead_factor,
            "context_length": context_length,
            "batch_size": batch_size,
            "safety_headroom": _SAFETY_HEADROOM,
        },
        "weights_gb": round(weights_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "total_gb": round(total_gb, 2),
        "required_gb_with_headroom": round(required_gb, 2),
        "fits_on": fits_on,
        "cheapest_fit": cheapest_fit,
        "caveats": caveats,
    }
