from server import reconciliation
from server.reconciliation import index_by_key, normalize_model_name


def test_strips_trailing_parenthetical():
    assert normalize_model_name("Claude Fable 5 (High)") == "claude fable 5"
    assert (
        normalize_model_name("Claude Fable 5 (Adaptive Reasoning, Max Effort, Extended Thinking)")
        == "claude fable 5"
    )


def test_permaslug_drops_vendor_prefix():
    assert (
        normalize_model_name("anthropic/claude-3.5-sonnet", is_permaslug=True)
        == "claude 3 5 sonnet"
    )


def test_plain_name_not_treated_as_permaslug():
    # No is_permaslug flag: a "/" in a plain display name is left alone rather
    # than being (wrongly) treated as a vendor prefix.
    assert normalize_model_name("gpt-4o") == "gpt 4o"


def test_empty_and_none_input():
    assert normalize_model_name(None) == ""
    assert normalize_model_name("") == ""


def test_overrides_applied_after_normalization(monkeypatch):
    monkeypatch.setitem(reconciliation.MODEL_NAME_OVERRIDES, "weird alias", "canonical name")
    assert normalize_model_name("Weird Alias") == "canonical name"


def test_index_by_key_drops_unkeyable_rows():
    rows = [{"model_name": "Alpha (High)"}, {"model_name": ""}, {"model_name": None}]
    index = index_by_key(rows, "model_name")
    assert list(index.keys()) == ["alpha"]
