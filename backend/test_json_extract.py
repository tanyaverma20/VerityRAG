"""
test_json_extract.py — json_extract.extract_json_object(), including a
real regression test built from an ACTUAL raw response captured from a
real vision-capable reasoning model (qwen/qwen3.6-27b via
GROQ_VISION_MODEL) during live verification — this exact response is what
exposed the original bug (a <think> preamble containing LaTeX curly
braces broke the old greedy-regex extractor).
"""
from json_extract import extract_json_object, _find_balanced_json_objects


def test_bare_json_no_preamble():
    assert extract_json_object('{"answer": "hi", "grounded": true}') == {"answer": "hi", "grounded": True}


def test_json_inside_markdown_fence():
    raw = 'Sure, here you go:\n```json\n{"answer": "hi", "grounded": true}\n```\nHope that helps!'
    assert extract_json_object(raw) == {"answer": "hi", "grounded": True}


def test_json_inside_bare_fence_no_json_tag():
    raw = '```\n{"answer": "hi"}\n```'
    assert extract_json_object(raw) == {"answer": "hi"}


def test_none_when_nothing_parses():
    assert extract_json_object("I don't know, sorry, no JSON here at all.") is None


def test_none_for_empty_string():
    assert extract_json_object("") is None
    assert extract_json_object(None) is None


def test_think_preamble_with_latex_braces_does_not_break_extraction():
    """The exact failure mode found live: a reasoning preamble containing
    LaTeX notation like \\frac{a}{b} has its own balanced { } pairs that
    have nothing to do with the answer — the old greedy regex spanned from
    one of those all the way to the real closing brace and produced
    invalid JSON. The real answer must still be extracted correctly."""
    raw = (
        "<think>\n"
        "This visualizes the formula $Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$.\n"
        "Let me also consider {some other reasoning fragment with braces}.\n"
        "Final answer coming up.\n"
        "</think>\n\n"
        '{"answer": "Scaled dot-product attention.", "grounded": true, "evidence_sufficient": true, "document_ids": []}'
    )
    result = extract_json_object(raw)
    assert result == {
        "answer": "Scaled dot-product attention.",
        "grounded": True,
        "evidence_sufficient": True,
        "document_ids": [],
    }


def test_real_captured_reasoning_model_response_with_fenced_json():
    """A real, verbatim-shaped raw response (condensed) matching what was
    actually captured from qwen/qwen3.6-27b during live Explain Figure
    verification — <think> preamble full of LaTeX braces, THEN the real
    answer in a ```json fence. This is the exact case that silently
    degraded every vision call to the text-only fallback before this fix."""
    raw = (
        "<think>\n"
        "This visualizes the formula $Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$.\n"
        "**Bottom Diagram:** Titled \"Multi-Head Attention\".\n"
        "MultiHead(Q, K, V) = Concat(head1, ...,headh)W^O where headi = Attention(QW^Q_i, KW^K_i, VW^V_i)\n"
        "Final check of the JSON structure.\n"
        "</think>\n\n"
        "```json\n"
        "{\n"
        '  "answer": "Figure 2 displays two diagrams illustrating attention mechanisms.",\n'
        '  "grounded": true,\n'
        '  "evidence_sufficient": true,\n'
        '  "document_ids": ["doc_1"]\n'
        "}\n"
        "```"
    )
    result = extract_json_object(raw)
    assert result is not None
    assert result["answer"] == "Figure 2 displays two diagrams illustrating attention mechanisms."
    assert result["grounded"] is True
    assert result["document_ids"] == ["doc_1"]


def test_multiple_balanced_objects_picks_the_last_one():
    """A preamble that itself contains an unrelated, syntactically valid
    JSON-shaped example must not be mistaken for the real answer — the
    real answer (the last one) must win."""
    raw = 'Example format: {"foo": "bar"}\n\nActual answer: {"answer": "real one", "grounded": true}'
    result = extract_json_object(raw)
    assert result == {"answer": "real one", "grounded": True}


def test_find_balanced_json_objects_ignores_braces_inside_strings():
    text = 'noise {"a": "text with a } brace inside a string"} more noise {"b": 2}'
    spans = _find_balanced_json_objects(text)
    assert len(spans) == 2
    assert spans[0].startswith('{"a"')
    assert spans[1] == '{"b": 2}'


def test_malformed_fence_falls_through_to_balanced_scan():
    """If the fenced content itself doesn't parse, extraction must still
    find a real JSON object elsewhere rather than giving up."""
    raw = '```json\nnot actually json\n```\n{"answer": "found via fallback"}'
    result = extract_json_object(raw)
    assert result == {"answer": "found via fallback"}


def test_never_raises_on_garbage_input():
    for garbage in ["{{{{{", "}}}}}", "{" * 500, '{"unterminated": ', "null", "42", "[1, 2, 3]"]:
        result = extract_json_object(garbage)
        assert result is None or isinstance(result, dict)
