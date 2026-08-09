"""
test_doc_titles.py — deterministic display-title resolution. Pure function
tests, no LLM, no retrieval, no server.
"""
from doc_titles import humanize_filename, resolve_display_title


def test_humanizes_real_filenames():
    assert humanize_filename("OS_Full_Notes.pdf") == "OS Full Notes"
    assert humanize_filename("DBMS_Full_Notes(1).PDF") == "DBMS Full Notes"
    assert humanize_filename("attention-is-all-you-need.pdf") == "Attention Is All You Need"
    assert humanize_filename("RetentionAI_Project_Documentation.pdf") == "RetentionAI Project Documentation"


def test_rejects_technical_filenames():
    assert humanize_filename("d926a10c1260ae70.pdf") is None
    assert humanize_filename("bdfaa68d8984f0dc.pdf") is None
    assert humanize_filename("550e8400-e29b-41d4-a716-446655440000.pdf") is None
    assert humanize_filename("scan001.pdf") is None
    assert humanize_filename("Untitled.pdf") is None
    assert humanize_filename("document.pdf") is None
    assert humanize_filename("12345.pdf") is None
    assert humanize_filename(None) is None
    assert humanize_filename("") is None


def test_resolve_prefers_filename_over_fallback_title():
    assert resolve_display_title("abc123", "OS_Full_Notes.pdf", fallback_title="Something Else") == "OS Full Notes"


def test_resolve_falls_back_to_llm_title_when_filename_meaningless():
    assert resolve_display_title("d926a10c1260ae70", "d926a10c1260ae70.pdf", fallback_title="Operating Systems Concepts") == "Operating Systems Concepts"


def test_resolve_never_returns_the_document_id():
    result = resolve_display_title("d926a10c1260ae70", "d926a10c1260ae70.pdf", fallback_title=None, index=0)
    assert result != "d926a10c1260ae70"
    assert "d926a10c1260ae70" not in result
    assert result == "Document 1"


def test_resolve_generic_fallback_without_index():
    # A document_id with genuinely no stored chunks anywhere (unlike
    # "bdfaa68d8984f0dc", which is attention.pdf's real content-hash id and
    # IS pre-ingested into the isolated test fixture — see
    # test_content_derivation_uses_the_documents_own_text below).
    assert resolve_display_title("no_such_document_id_at_all", None, fallback_title=None) == "Untitled Document"


def test_content_derivation_uses_the_documents_own_text():
    # "bdfaa68d8984f0dc" is attention.pdf's real content-hash document_id,
    # pre-ingested into the isolated test Chroma fixture by conftest.py.
    # With no usable filename or fallback_title, resolve_display_title must
    # derive a real topic from the paper's own stored chunk text rather than
    # falling back to a generic "Untitled Document" label.
    result = resolve_display_title("bdfaa68d8984f0dc", None, fallback_title=None)
    assert result != "Untitled Document"
    assert "bdfaa68d8984f0dc" not in result
    assert len(result.split()) <= 5


def test_resolve_rejects_a_fallback_title_that_is_itself_just_the_hash():
    # If the model echoed the id/filename back as "title", that's not a real
    # title either — must not leak through.
    result = resolve_display_title("d926a10c1260ae70", None, fallback_title="d926a10c1260ae70", index=2)
    assert result == "Document 3"
