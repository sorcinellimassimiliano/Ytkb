"""Unit tests for the binding citation validator (no DB)."""

from __future__ import annotations

from ytkb.knowledge.merge import cited_unit_ids, validate_article


def test_cited_unit_ids_parses_all():
    md = "Frase uno [unit:1]. Frase due [unit:42] e [unit:7]."
    assert cited_unit_ids(md) == {1, 7, 42}


def test_valid_article_passes():
    md = "- Tecnica [unit:1]\n- Strumento [unit:2]\n"
    assert validate_article(md, allowed_ids={1, 2, 3}, required_ids={1, 2}) == []


def test_invalid_citation_flagged():
    md = "- Frase [unit:99]\n"
    errors = validate_article(md, allowed_ids={1, 2}, required_ids={1})
    assert any("non valide" in e for e in errors)


def test_missing_new_unit_flagged():
    md = "- Solo la vecchia [unit:1]\n"
    errors = validate_article(md, allowed_ids={1, 2}, required_ids={2})
    assert any("non citate" in e for e in errors)


def test_missing_allowed_when_explicitly_omitted():
    md = "- Vecchia [unit:1]\n\nUnità omesse: [unit:2] perché ridondante.\n"
    # required 2 is not cited as a claim, but the omission note is present.
    assert validate_article(md, allowed_ids={1, 2}, required_ids={2}) == []
