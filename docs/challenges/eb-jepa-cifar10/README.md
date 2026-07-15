# Exercices EB-JEPA CIFAR-10 — Mode d'emploi

## Racine des chemins

Tous les chemins `src/...`, `configs/...` et `scripts/...` partent de :

```text
eb_jepa_cifar10_comparison/
```

Par exemple, `src/comparison/model.py` désigne réellement
`eb_jepa_cifar10_comparison/src/comparison/model.py` depuis la racine du dépôt.

## Comment résoudre un exercice

1. Lis `Où travailler` et ouvre uniquement les fichiers indiqués.
2. Lis `Contexte` avant le code : il explique ce qui existe déjà.
3. Recopie le `Code de départ` dans le fichier indiqué si le symbole n'existe pas.
4. Remplace les `...` en suivant `Comportement attendu`.
5. Lance la `Vérification manuelle` depuis `eb_jepa_cifar10_comparison/`.
6. Compare le résultat avec la checklist `Définition de terminé`.
7. Consulte les indices dans l'ordre seulement si tu bloques.

Les commandes supposent que le projet est installé avec `pip install -e .` ou
que `PYTHONPATH=src` est défini.

## Deux types d'exercices

### Copie guidée

La solution existe dans `external/eb_jepa/` ou `Predictor/`. Tu dois retrouver
et copier les symboles indiqués. L'énoncé liste exhaustivement les modifications
autorisées ; ne refactorise pas les équations pendant la copie.

### Conception

Le composant n'existe pas tel quel dans la référence. Tu reçois son contexte,
son contrat, un exemple et un squelette, mais pas son corps complet.

## Absence de tests automatisés

Ce projet ne demande ni `pytest`, ni dossier `tests/`. Les blocs de vérification
sont des contrôles ponctuels destinés à te donner un retour immédiat. Ils ne font
pas partie du code livré.

## Conventions de tenseurs

- `B` : taille du batch ;
- images : `(B, 3, 32, 32)` ;
- features du backbone : `(B, 512)` ;
- représentations des têtes : `(B, 2048)` ;
- une vue est une augmentation aléatoire d'une image source.

## Aide

Chaque exercice contient trois indices progressifs. L'indice 1 rappelle le
concept, l'indice 2 indique la structure à employer, et l'indice 3 décrit presque
l'algorithme sans fournir le code final.
