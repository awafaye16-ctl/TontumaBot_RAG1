from intent.router import detect_intent


def test_procedure_documents():
    assert detect_intent("Quels sont les documents nécessaires pour une carte d'identité ?") == "procedure"


def test_procedure_comment_faire():
    assert detect_intent("Comment créer une entreprise au Sénégal ?") == "procedure"
    assert detect_intent("Comment renouveler ma carte d'identité ?") == "procedure"


def test_orientation_lieu():
    assert detect_intent("Où dois-je déposer cette demande ?") == "orientation"


def test_orientation_service():
    assert detect_intent("Quel service s'occupe de cette procédure ?") == "orientation"
    assert detect_intent("Quelle administration dois-je contacter ?") == "orientation"


def test_orientation_cout():
    assert detect_intent("Combien coûte un passeport ?") == "orientation"
    assert detect_intent("Quel est le prix de la carte d'identité ?") == "orientation"


def test_no_false_positive_sur_mot_ou():
    # "ou" dans "documents" ne doit PAS déclencher orientation
    assert detect_intent("Quels documents pour un extrait de naissance ou un acte de mariage ?") == "procedure"