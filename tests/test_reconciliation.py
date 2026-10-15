from server import reconciliation
from server.reconciliation import index_by_key, normalize_model_name


def test_strips_trailing_parenthetical():
    # Words are sorted (see test_word_order_insensitive below), so compare
    # against the sorted form rather than the original word order.
    assert normalize_model_name("Claude Fable 5 (High)") == "5 claude fable"
    assert (
        normalize_model_name("Claude Fable 5 (Adaptive Reasoning, Max Effort, Extended Thinking)")
        == "5 claude fable"
    )


def test_permaslug_drops_vendor_prefix():
    # "3.5" stays one token (dots aren't a separator) rather than fragmenting
    # into "3"/"5", which would collide with e.g. "3.1" or "5.0" releases.
    assert (
        normalize_model_name("anthropic/claude-3.5-sonnet", is_permaslug=True)
        == "3.5 claude sonnet"
    )


def test_decimal_versions_dont_collide():
    # The actual round-2 bug: naive tokenization degraded both "3.3" and "3.1"
    # to a shared bare "3", so a 3.3 query matched 3.1 rows. Confirm distinct
    # versions now normalize to genuinely different keys.
    assert normalize_model_name("Llama 3.3 Instruct 70B") != normalize_model_name(
        "Llama 3.1 Instruct 70B"
    )


def test_hyphenated_version_in_slug_merges_to_decimal():
    # A pasted slug-like string ("llama-3-3-70b") should tokenize the same as
    # its dotted display form.
    from server.reconciliation import tokenize_model_name

    assert tokenize_model_name("llama-3-3-70b") == ["llama", "3.3", "70b"]


def test_plain_name_not_treated_as_permaslug():
    # No is_permaslug flag: a "/" in a plain display name is left alone rather
    # than being (wrongly) treated as a vendor prefix.
    assert normalize_model_name("gpt-4o") == "4o gpt"


def test_empty_and_none_input():
    assert normalize_model_name(None) == ""
    assert normalize_model_name("") == ""


def test_overrides_applied_after_normalization(monkeypatch):
    # The lookup key must already be in sorted-token form, same as what
    # normalize_model_name produces before consulting the override map.
    monkeypatch.setitem(reconciliation.MODEL_NAME_OVERRIDES, "alias weird", "canonical name")
    assert normalize_model_name("Weird Alias") == "canonical name"


def test_word_order_insensitive():
    # Confirmed against live data: the same model can appear in a different
    # word order per source (AA's "Llama 3.3 Instruct 70B" vs. a permaslug-
    # derived "llama 3.3 70b instruct") — plain ordered-string equality missed
    # this; sorting the words fixes it.
    assert normalize_model_name("Llama 3.3 Instruct 70B") == normalize_model_name(
        "meta-llama/llama-3.3-70b-instruct", is_permaslug=True
    )


def test_index_by_key_drops_unkeyable_rows():
    rows = [{"model_name": "Alpha (High)"}, {"model_name": ""}, {"model_name": None}]
    index = index_by_key(rows, "model_name")
    assert list(index.keys()) == ["alpha"]


def test_best_token_match_does_not_confuse_adjacent_versions():
    from server.reconciliation import best_token_match

    candidates = [
        ("Llama 3.1 Instruct 70B", "3.1"),
        ("Llama 3.3 Instruct 70B", "3.3"),
        ("Hermes 3 - Llama-3.1 70B", "hermes"),
    ]
    result = best_token_match("llama 3.3 70b", candidates)
    assert result is not None
    payload, matched_name, others = result
    assert payload == "3.3"
    assert matched_name == "Llama 3.3 Instruct 70B"


def test_best_token_match_strips_permaslug_vendor_prefix():
    from server.reconciliation import best_token_match

    candidates = [("meta-llama/llama-3.3-70b-instruct", "row")]
    result = best_token_match("llama 3.3 70b", candidates, is_permaslug=True)
    assert result is not None
    assert result[0] == "row"


def test_best_token_match_returns_none_when_nothing_qualifies():
    from server.reconciliation import best_token_match

    assert best_token_match("totally unrelated model", [("Llama 3.3", "x")]) is None
