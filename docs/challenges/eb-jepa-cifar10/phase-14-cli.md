# Phase 14 — Écrire les quatre entrées CLI

Crée quatre scripts minces. Toute logique scientifique reste sous `src/comparison`.

### Exercice 14.1 — `Scripts train, evaluate, benchmark et report`

**Difficulté :** Moyen

#### Où travailler

`scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py` et `scripts/report.py`.

#### Contexte

Les scripts offrent une interface reproductible. Toute logique reste dans `src/comparison` afin qu'un import Python et un appel CLI utilisent le même protocole.

#### Objectif

Écrire quatre fonctions `main()` avec argparse et un bloc d'entrée standard.

#### Entrées

Train : config, seed, output, resume, overrides. Evaluate : config, checkpoint, cinq pair seeds, output. Benchmark : config, checkpoint, output. Report : six JSON et deux outputs.

#### Sorties

Code de sortie 0 et chemin du fichier produit affiché.

#### Comportement attendu

- analyser seulement les arguments.
- charger la configuration avec load_config.
- appeler exactement une fonction métier.
- afficher le résultat.
- protéger l'appel par le bloc main.

#### Exemple

`python scripts/train.py --config configs/baseline.yaml --seed 1 --output-dir runs/baseline/1 optimization.epochs=1` appelle `comparison.train.run`.

#### Contraintes

- quatre scripts.
- aucune boucle d'entraînement dupliquée.
- aucun fallback après OOM.
- architecture lue dans la config.

#### Code de départ

```python
# scripts/train.py
import argparse
from comparison.config import load_config
from comparison.train import run


def main() -> None:
    ...


if __name__ == "__main__":
    main()

# Reprends cette structure pour evaluate.py, benchmark.py et report.py.
```

#### Étapes conseillées

1. définis les quatre parsers.
2. charge et valide les arguments.
3. appelle les fonctions métier.
4. affiche les chemins.

#### Vérification manuelle

Les quatre commandes `python scripts/<nom>.py --help` doivent fonctionner sans télécharger CIFAR ni initialiser CUDA.

#### Définition de terminé

- [ ] quatre aides fonctionnent.
- [ ] scripts minces.
- [ ] invocations documentées.
- [ ] aucune logique scientifique dupliquée.

#### Indices

1. `nargs='*'` accepte les overrides pointés.
2. Importer un script n'exécute pas main.
3. Toutes les validations importantes existent déjà sous src.

