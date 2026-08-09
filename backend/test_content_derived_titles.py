"""
test_content_derived_titles.py — proves the final display-title fallback is
a real, deterministic topic pulled from the document's OWN extracted
content (never "Untitled Document", never an internal id), with zero LLM
calls — see doc_titles.derive_topic_from_content / resolve_display_title.
"""
import uuid

import config
from ingest import get_collection
from doc_titles import derive_topic_from_content, resolve_display_title


def _push_chunk(document_id: str, suffix: str, text: str, page_number: int = 1) -> None:
    col = get_collection()
    chunk_id = f"doc_{document_id}_{suffix}"
    col.add(
        documents=[text],
        ids=[chunk_id],
        metadatas=[{
            "document_id": document_id, "filename": f"{document_id}.pdf", "source": f"{document_id}.pdf",
            "page_number": page_number, "section": "Body", "chunk_id": chunk_id,
            "parent_id": f"{document_id}_{suffix}_parent", "chunk_type": "child",
        }],
    )


def _fresh_doc_id() -> str:
    assert config.COLLECTION_NAME == "test_collection", "Must run inside the isolated Phase 8 fixture"
    return "content_topic_" + uuid.uuid4().hex[:12]


def test_derives_the_repeated_multi_word_topic_from_content():
    doc_id = _fresh_doc_id()
    _push_chunk(doc_id, "c1", "Introduction to Database Management System concepts and terminology.", page_number=1)
    _push_chunk(doc_id, "c2", "A Database Management System organizes data into related tables.", page_number=1)
    _push_chunk(doc_id, "c3", "This chapter covers indexing, transactions, and the Database Management System architecture.", page_number=2)

    topic = derive_topic_from_content(doc_id)
    assert topic is not None
    assert "Database Management System" in topic or topic == "Database Management System"
    assert doc_id not in topic


def test_resolve_display_title_uses_content_when_filename_is_a_hash():
    doc_id = _fresh_doc_id()
    hash_filename = uuid.uuid4().hex + ".pdf"  # a pure hex filename, like the real d926a10c1260ae70.pdf case
    _push_chunk(doc_id, "c1", "Operating System schedulers decide which process runs next.", page_number=1)
    _push_chunk(doc_id, "c2", "Every Operating System must manage memory, processes, and I/O.", page_number=1)
    _push_chunk(doc_id, "c3", "The Operating System kernel mediates access to hardware resources.", page_number=2)

    result = resolve_display_title(doc_id, filename=hash_filename, fallback_title=None)
    assert result != "Untitled Document"
    assert doc_id not in result
    assert "Operating System" in result


def test_single_occurrence_phrase_is_not_trusted():
    # A phrase that appears only once isn't reliable enough to promote to a
    # document topic — must not overfit to noise.
    doc_id = _fresh_doc_id()
    _push_chunk(doc_id, "c1", "A brief mention of Quantum Cryptography appears here once.", page_number=1)
    topic = derive_topic_from_content(doc_id)
    assert topic != "Quantum Cryptography"


def test_falls_back_to_repeated_single_word_when_no_multiword_phrase_repeats():
    doc_id = _fresh_doc_id()
    _push_chunk(doc_id, "c1", "Segmentation is discussed at length in this section of the notes.", page_number=1)
    _push_chunk(doc_id, "c2", "Segmentation avoids internal fragmentation in memory allocation.", page_number=1)
    _push_chunk(doc_id, "c3", "Segmentation partitions memory into logical, variable-size units.", page_number=2)

    topic = derive_topic_from_content(doc_id)
    assert topic == "Segmentation"


def test_no_stored_chunks_returns_none_not_a_guess():
    assert derive_topic_from_content("truly_nonexistent_document_id") is None


def test_never_returns_document_id_even_when_content_is_unusable():
    doc_id = _fresh_doc_id()
    hash_filename = uuid.uuid4().hex + ".pdf"
    _push_chunk(doc_id, "c1", "12345 67890 00000 11111 22222 33333.", page_number=1)  # no capitalized words at all
    result = resolve_display_title(doc_id, filename=hash_filename, fallback_title=None, index=0)
    assert result == "Document 1"
    assert doc_id not in result
