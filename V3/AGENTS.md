# Vérification locale

Depuis V3, sous Git Bash/Windows :

- Tests de non-régression : `.venv/Scripts/python -X utf8 -m unittest discover -s tests -p 'test_*regressions.py'`.
- Audit hors ligne des sources réellement configurées : `.venv/Scripts/python -X utf8 download_models.py --check`.
- L'audit vérifie les fichiers requis dans les dossiers locaux et la révision `main` du cache Hugging Face. Il ne prouve ni le chargement, ni l'inférence, ni la disponibilité ou les quotas des API.
- Les tests API utilisent TestClient sans démarrer le lifespan pour ne pas indexer la base réelle.
- Ne pas afficher les clés de `.env`, supprimer les poids ni purger ChromaDB pour résoudre un défaut de disponibilité.
