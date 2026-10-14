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
    assert (
        normalize_model_name("anthropic/claude-3.5-sonnet", is_permaslug=True)
        == "3 5 claude sonnet"
    )


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
