# EB-JEPA CIFAR-10 — Roadmap d'implémentation personnelle

Cette roadmap est l'index du projet. Les énoncés complets vivent dans
`docs/challenges/eb-jepa-cifar10/` : chaque exercice y est présenté comme un
problème LeetCode autonome appliqué à un véritable dépôt Python.

## But scientifique

Comparer deux têtes de représentation sur CIFAR-10 en gardant tout le reste
identique :

```text
baseline : image -> ResNet-18 -> MLP officiel -> représentation 2048
cortical : image -> ResNet-18 -> FixedTreePredictor -> représentation 2048
```

L'exemple Image d'EB-JEPA est une JEA utilisant VICReg ; il ne contient pas de
prédicteur contextuel I-JEPA. Le score principal est la loss VICReg sur le split
test, accompagnée d'un diagnostic d'effondrement. Aucun label et aucun linear
probe ne participent au protocole.

## Référence figée

- dépôt : `https://github.com/facebookresearch/eb_jepa.git` ;
- clone intact : `external/eb_jepa/` ;
- commit : `966e61e9285b3a876f49b9774e9720d9a99a7925` ;
- licence : Apache-2.0 ;
- projet à modifier : `eb_jepa_cifar10_comparison/` ;
- registre de provenance : `eb_jepa_cifar10_comparison/SOURCES.md`.

Ne modifie jamais `external/eb_jepa/`.

## Protocole commun imposé

- CIFAR-10 : 45 000 images train, 5 000 validation, 10 000 test ;
- ResNet-18 adapté aux images 32×32, sortie 512 ;
- têtes 512 vers 2048 ; batch size 256 ; 300 epochs ;
- seeds 1, 1000 et 10000 ; LARS ; LR 0,3 ; warmup 10 epochs ;
- weight decay `1e-4` ; VICReg `(1, 1, 80)` ; précision BF16 ;
- test évalué seulement après le gel de la configuration ;
- benchmark final sur A100 ;
- aucun test automatisé, dossier `tests/` ou usage de `pytest`.

## Parcours

Commence par lire le [mode d'emploi](../../challenges/eb-jepa-cifar10/README.md),
puis réalise les phases dans l'ordre :

| Phase | Sujet | Exercices |
|---:|---|---:|
| 0 | [Projet autonome](../../challenges/eb-jepa-cifar10/phase-00-project-setup.md) | 3 |
| 1 | [Augmentations officielles](../../challenges/eb-jepa-cifar10/phase-01-augmentations.md) | 6 |
| 2 | [Dataset et splits](../../challenges/eb-jepa-cifar10/phase-02-data.md) | 6 |
| 3 | [Backbone et baseline](../../challenges/eb-jepa-cifar10/phase-03-backbone-baseline.md) | 2 |
| 4 | [Prédicteur cortical](../../challenges/eb-jepa-cifar10/phase-04-cortical-predictor.md) | 8 |
| 5 | [Modèle commun](../../challenges/eb-jepa-cifar10/phase-05-model.md) | 3 |
| 6 | [VICReg et anti-effondrement](../../challenges/eb-jepa-cifar10/phase-06-vicreg.md) | 4 |
| 7 | [LARS et scheduler](../../challenges/eb-jepa-cifar10/phase-07-optimization.md) | 2 |
| 8 | [Configuration](../../challenges/eb-jepa-cifar10/phase-08-configuration.md) | 3 |
| 9 | [Utilitaires et checkpoints](../../challenges/eb-jepa-cifar10/phase-09-checkpoints.md) | 3 |
| 10 | [Entraînement](../../challenges/eb-jepa-cifar10/phase-10-training.md) | 4 |
| 11 | [Évaluation test](../../challenges/eb-jepa-cifar10/phase-11-evaluation.md) | 5 |
| 12 | [Benchmark A100](../../challenges/eb-jepa-cifar10/phase-12-benchmark-a100.md) | 5 |
| 13 | [Rapport et décision](../../challenges/eb-jepa-cifar10/phase-13-report.md) | 4 |
| 14 | [Scripts CLI](../../challenges/eb-jepa-cifar10/phase-14-cli.md) | 1 |

Total : **59 exercices**.

## Arborescence cible

```text
eb_jepa_cifar10_comparison/
├── LICENSE.md
├── README.md
├── SOURCES.md
├── pyproject.toml
├── configs/{baseline,cortical}.yaml
├── src/comparison/
│   ├── __init__.py
│   ├── augmentation.py
│   ├── data.py
│   ├── backbone.py
│   ├── heads.py
│   ├── model.py
│   ├── losses.py
│   ├── optim.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── report.py
│   └── cortical/
└── scripts/{train,evaluate,benchmark,report}.py
```

## Définition globale de réussite

Les six runs utilisent le même backbone, les mêmes données et les mêmes
hyperparamètres. La tête corticale réussit si son score test VICReg reste dans la
tolérance fixée face au baseline tout en réduisant le compute mesuré sur A100.
Le rapport doit aussi montrer l'absence d'effondrement et les résultats par seed.
