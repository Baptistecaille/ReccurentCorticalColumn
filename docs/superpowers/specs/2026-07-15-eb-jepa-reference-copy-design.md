# Conception — Référence EB-JEPA et projet de comparaison CIFAR-10

## Objectif

Cloner le dépôt officiel `facebookresearch/eb_jepa` comme référence locale
immuable, puis créer un projet indépendant contenant uniquement les fonctions
EB-JEPA nécessaires pour comparer la tête officielle et le
`FixedTreePredictor` sur CIFAR-10 avec VICReg.

## Structure retenue

```text
ReccurentCorticalColumn/
├── external/
│   └── eb_jepa/                 # clone officiel intact
├── eb_jepa_cifar10_comparison/  # projet expérimental autonome
│   ├── src/
│   ├── configs/
│   ├── scripts/
│   ├── SOURCES.md
│   └── pyproject.toml
└── Predictor/                   # implémentation corticale existante
```

Le dépôt placé dans `external/eb_jepa/` ne reçoit aucune modification. Son URL,
sa branche et son commit exact sont enregistrés dans `SOURCES.md`.

## Audit du dépôt officiel

Après le clonage, chaque brique utile est classée dans l’une de ces catégories :

1. fonction réutilisable sans modification ;
2. fonction à copier puis adapter explicitement à CIFAR-10 ;
3. fonction absente du dépôt et à écrire dans le projet de comparaison ;
4. fonction inutile pour le protocole retenu.

L’audit couvre le pipeline image, les augmentations, le backbone, la tête de
référence, VICReg, l’optimiseur, le scheduler, les checkpoints, l’entraînement,
l’évaluation et les mesures de compute.

## Règles de copie et d’attribution

Une fonction officielle n’est copiée que si un import direct rendrait le projet
expérimental dépendant de la structure interne du clone ou empêcherait son
exécution autonome. Chaque fichier repris comporte un commentaire indiquant :

- le dépôt d’origine ;
- le chemin du fichier source ;
- le commit source ;
- les adaptations locales effectuées.

`SOURCES.md` fournit la table complète entre symboles locaux et symboles
officiels. La licence du dépôt officiel est inspectée avant toute copie et ses
obligations sont conservées dans le nouveau projet.

## Architecture de la comparaison

Les deux systèmes utilisent exactement les mêmes composants pour CIFAR-10 :

- split train, validation et test ;
- génération des deux vues ;
- backbone et dimension de features ;
- loss VICReg et diagnostic d’effondrement ;
- optimiseur, scheduler, précision et nombre d’epochs ;
- boucle d’entraînement, checkpoints et évaluation ;
- seeds et procédure de benchmark A100.

Seule la tête varie. La variante de référence utilise la tête effectivement
fournie ou prescrite par EB-JEPA après audit du dépôt. La variante corticale
utilise `FixedTreePredictor` derrière un adaptateur minimal, sans projection
entraînable supplémentaire destinée à améliorer artificiellement son score.

## Adaptation CIFAR-10

Les adaptations nécessaires aux images 32×32 sont isolées et documentées : stem
du backbone, crop scale, augmentations, tailles de batch et chemins du dataset.
Le split test ne sert jamais à choisir une configuration. Aucun label n’entre
dans l’entraînement ou dans le score principal.

## Documentation d’implémentation

La roadmap existante est réécrite après l’audit. Chaque exercice contient :

- un paragraphe `Problème` au format LeetCode/Deep-ML ;
- le chemin exact du fichier à créer ou modifier ;
- l’origine du code lorsque la fonction vient d’EB-JEPA ;
- la signature attendue ;
- les adaptations autorisées ;
- le résultat observable attendu.

Les exercices ne demandent pas de réimplémenter une fonction officielle déjà
sélectionnée : ils demandent de la retrouver, la copier avec attribution, puis
d’effectuer uniquement les adaptations documentées.

## Contrôles d’exécution

Aucune suite de tests automatisés, aucun dossier `tests/` et aucune dépendance
`pytest` ne sont ajoutés. Les seuls contrôles demandés sont :

- chargement des configurations ;
- vérification des formes et valeurs finies à l’exécution ;
- deux batches courts pour chaque tête avant utilisation de l’A100 ;
- diagnostic d’effondrement ;
- reprise d’un checkpoint ;
- vérification de l’environnement et des mesures du benchmark.

## Critères de réussite

Le projet de comparaison est autonome, chaque fonction copiée est traçable vers
un commit officiel, et les deux prédicteurs passent par le même protocole. Le
rapport final compare le score test sans labels, l’effondrement, les paramètres,
les FLOPs, la mémoire et la latence sur A100.
