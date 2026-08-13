from language.detector import detect_language


def test_detect_wolof_basic():
    assert detect_language("dama beug wout kayitu juddu?") == "wo"


def test_detect_wolof_variants():
    for q in [
        "Fan lañuy def demande bi ?",
        "Jërejëf lool ci dimbal bi.",
        "Dama soxla doctoor ndax sama biir dafay metti lool.",
    ]:
        assert detect_language(q) == "wo", q


def test_detect_french_basic():
    assert detect_language("Comment obtenir un extrait de naissance ?") == "fr"


def test_detect_french_variants():
    for q in [
        "Quels sont les documents nécessaires pour un passeport ?",
        "Où dois-je déposer cette demande ?",
        "Merci beaucoup pour votre aide.",
    ]:
        assert detect_language(q) == "fr", q


def test_detect_empty_string():
    assert detect_language("") == "fr"


def test_detect_mixed_with_wolof_loanword():
    # Un seul mot wolof suffit à classer la phrase comme wolof
    assert detect_language("Ana lañuy def ngir jël passeport ?") == "wo"