# Phase 0 — Préparer le projet autonome

Cette phase crée le paquet Python indépendant dans lequel toute la comparaison sera écrite.

### Exercice 0.1 — `Licence et provenance`

**Difficulté :** Facile

#### Où travailler

`LICENSE.md` et `SOURCES.md`. Source : `external/eb_jepa/LICENSE.md` et les fichiers listés dans le registre.

#### Contexte

Le projet réutilise du code Apache-2.0. Une personne doit pouvoir retrouver le dépôt, le commit et le symbole source sans connaître cette conversation.

#### Objectif

Copier la licence officielle sans la modifier et créer une ligne de provenance uniquement lorsqu'un symbole existe réellement dans le projet.

#### Entrées

Aucune entrée runtime.

#### Sorties

Deux documents Markdown ; `LICENSE.md` identique à la référence et un registre lisible.

#### Comportement attendu

- copier `external/eb_jepa/LICENSE.md` vers la racine du projet.
- inscrire le commit `966e61e9285b3a876f49b9774e9720d9a99a7925` dans `SOURCES.md`.
- pour chaque copie, donner symbole local, chemin source et adaptations.

#### Exemple

`VICRegLoss | src/comparison/losses.py | eb_jepa/losses.py | copié, imports adaptés` permet de retrouver l'origine de la classe.

#### Contraintes

- ne modifie pas le texte de la licence.
- n'annonce pas un symbole qui n'existe pas encore.
- indique chaque fichier local modifié.

#### Code de départ

```python
# SOURCES.md

| Symbole local | Fichier local | Source | Adaptations |
|---|---|---|---|
| ... | ... | ... | ... |
```

#### Étapes conseillées

1. commence par la licence.
2. ajoute les métadonnées du dépôt.
3. ajoute une ligne au registre après chaque copie.

#### Vérification manuelle

`cmp LICENSE.md ../external/eb_jepa/LICENSE.md` doit terminer sans sortie.

#### Définition de terminé

- [ ] la licence est identique octet par octet.
- [ ] le commit figé est présent.
- [ ] chaque ligne décrit une copie réelle.

#### Indices

1. Une licence se copie comme un artefact, pas comme du code à reformuler.
2. Le registre est une table Markdown.
3. Une ligne correspond à un symbole ou groupe de symboles provenant du même emplacement.

---

### Exercice 0.2 — `Déclarer le package dans pyproject.toml`

**Difficulté :** Facile

#### Où travailler

`pyproject.toml` à la racine du projet de comparaison.

#### Contexte

Le code vit sous `src/comparison`. Sans configuration de packaging, Python peut importer accidentellement un autre dossier `comparison` ou ne pas trouver le package.

#### Objectif

Déclarer un projet Python 3.12 installable avec setuptools et uniquement les six dépendances nécessaires.

#### Entrées

Aucune entrée runtime ; le fichier est lu par un outil de packaging.

#### Sorties

Un document TOML valide nommé `eb-jepa-cifar10-comparison`.

#### Comportement attendu

- définir `[build-system]` avec setuptools.
- définir `[project]`, Python `==3.12.*` et les dépendances.
- configurer la découverte des packages sous `src`.

#### Exemple

`pip install -e .` doit rendre `import comparison` possible depuis un autre dossier.

#### Contraintes

- dépendances : torch, torchvision, omegaconf, pyyaml, tqdm, fvcore.
- pas de pytest, wandb ou dépendances vidéo.
- conserver le layout `src/`.

#### Code de départ

```python
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eb-jepa-cifar10-comparison"
requires-python = "==3.12.*"
dependencies = [
    # À compléter avec exactement six dépendances.
]

[tool.setuptools.packages.find]
where = ["src"]
```

#### Étapes conseillées

1. écris d'abord les métadonnées.
2. ajoute les six bibliothèques runtime.
3. configure la découverte setuptools.

#### Vérification manuelle

`python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['name'])"` affiche `eb-jepa-cifar10-comparison`.

#### Définition de terminé

- [ ] le TOML se charge.
- [ ] six dépendances runtime sont présentes.
- [ ] les packages sont cherchés sous src.

#### Indices

1. TOML utilise des sections entre crochets.
2. `requires-python` est distinct des dépendances.
3. Setuptools doit connaître la racine du layout.

---

### Exercice 0.3 — `Exports publics paresseux`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/__init__.py` et `src/comparison/cortical/__init__.py`.

#### Contexte

Les scripts importeront quelques symboles stables, mais plusieurs modules n'existent pas encore. Un import eager de tous les modules ferait échouer `import comparison` dès la phase 0.

#### Objectif

Définir `__all__` et résoudre les symboles seulement lorsqu'ils sont demandés.

#### Entrées

Un nom d'attribut tel que `ImageSSL` reçu par `__getattr__`.

#### Sorties

Le symbole importé ou une exception `AttributeError` pour un nom inconnu.

#### Comportement attendu

- déclarer les cinq noms publics du package principal.
- associer chaque nom à son module dans `__getattr__`.
- exposer seulement `FixedTreePredictor` dans le sous-package cortical.

#### Exemple

`import comparison` fonctionne maintenant ; `comparison.ImageSSL` ne sera résolu qu'une fois `comparison.model` créé.

#### Contraintes

- aucune fausse classe temporaire.
- un nom inconnu lève AttributeError.
- les imports de modules utilisent importlib ou des imports locaux.

#### Code de départ

```python
from typing import Any

__all__ = ["ImageSSL", "build_head", "build_model", "VICRegLoss", "load_config"]


def __getattr__(name: str) -> Any:
    ...
```

#### Étapes conseillées

1. écris la table nom vers module.
2. importe le module demandé dans la fonction.
3. retourne son attribut ou lève AttributeError.

#### Vérification manuelle

`PYTHONPATH=src python -c "import comparison; print(comparison.__all__)"` affiche les cinq noms sans importer PyTorch.

#### Définition de terminé

- [ ] le package s'importe.
- [ ] la liste publique est exacte.
- [ ] le sous-package n'expose que le prédicteur.

#### Indices

1. `__all__` documente l'API, il ne charge rien seul.
2. `__getattr__` au niveau module est appelé pour un attribut absent.
3. Une table évite une longue chaîne de conditions.

