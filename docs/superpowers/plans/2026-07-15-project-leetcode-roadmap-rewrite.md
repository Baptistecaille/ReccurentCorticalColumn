# Project LeetCode Roadmap Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réécrire les 59 exercices des phases 0 à 14 sous forme de problèmes autonomes « LeetCode appliqué à un projet ».

**Architecture:** Le fichier de roadmap principal devient l'index du parcours et conserve les règles scientifiques globales. Les énoncés détaillés sont répartis dans un fichier Markdown par phase sous `docs/challenges/eb-jepa-cifar10/`, ce qui permet d'ouvrir un groupe d'exercices sans charger toute la roadmap.

**Tech Stack:** Markdown, Python 3.12, PyTorch, torchvision, OmegaConf, CIFAR-10, VICReg.

## Global Constraints

- Aucun test automatisé, dossier `tests/` ou usage de `pytest`.
- Chaque exercice contient les quatorze rubriques définies dans `docs/superpowers/specs/2026-07-15-project-leetcode-roadmap-design.md`.
- Les chemins locaux sont relatifs à `eb_jepa_cifar10_comparison/` et répétés dans chaque exercice.
- Les fonctions à concevoir gardent leur solution cachée ; les fonctions à copier indiquent toutes les adaptations autorisées.
- Les vérifications utilisent seulement de petites commandes ou scripts manuels.
- Le protocole scientifique et les hyperparamètres de comparaison ne changent pas.

---

### Task 1: Index et gabarit commun

**Files:**
- Modify: `docs/superpowers/plans/2026-07-15-eb-jepa-cifar10-self-implementation-roadmap.md`
- Create: `docs/challenges/eb-jepa-cifar10/README.md`

**Interfaces:**
- Consumes: la spécification pédagogique validée et les règles scientifiques de la roadmap actuelle.
- Produces: un index vers les quinze phases et un rappel du gabarit commun.

- [ ] Remplacer les énoncés condensés du fichier principal par un index de phases.
- [ ] Conserver l'objectif, l'arborescence, les règles scientifiques, la configuration et l'ordre de réalisation.
- [ ] Ajouter dans le README la convention des chemins, la manière d'utiliser un exercice et la légende des exercices de copie/conception.
- [ ] Vérifier manuellement tous les liens Markdown locaux avec `rg` et un contrôle de présence des fichiers.

### Task 2: Phases 0 à 4 — socle et architectures

**Files:**
- Create: `docs/challenges/eb-jepa-cifar10/phase-00-project-setup.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-01-augmentations.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-02-data.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-03-backbone-baseline.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-04-cortical-predictor.md`

**Interfaces:**
- Consumes: les fichiers locaux présents, `external/eb_jepa` et `Predictor/`.
- Produces: 25 problèmes autonomes couvrant le packaging, les données et les deux architectures.

- [ ] Écrire chaque exercice avec les quatorze rubriques obligatoires.
- [ ] Pour les copies, énumérer symbole source, destination et adaptations autorisées.
- [ ] Pour `CorticalHead` et `build_head`, expliquer les dépendances déjà disponibles et fournir un squelette local complet.
- [ ] Ajouter des vérifications manuelles de formes ne nécessitant ni dataset téléchargé ni GPU.

### Task 3: Phases 5 à 9 — modèle, loss et infrastructure

**Files:**
- Create: `docs/challenges/eb-jepa-cifar10/phase-05-model.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-06-vicreg.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-07-optimization.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-08-configuration.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-09-checkpoints.md`

**Interfaces:**
- Consumes: `ResNet18`, les têtes, les loaders et les paramètres de configuration.
- Produces: 15 problèmes autonomes pour le modèle commun, VICReg, LARS, le scheduler, OmegaConf et la reprise.

- [ ] Placer `ImageSSL`, `build_model` et `count_parameters` dans `src/comparison/model.py`.
- [ ] Définir chaque terme VICReg et chaque forme avant le code de départ.
- [ ] Expliquer `DictConfig`, le warmup et l'état d'un checkpoint dans leur contexte d'utilisation.
- [ ] Fournir des exemples reproductibles sur CPU et des vérifications manuelles sans tests.

### Task 4: Phases 10 à 14 — expérience et décision finale

**Files:**
- Create: `docs/challenges/eb-jepa-cifar10/phase-10-training.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-11-evaluation.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-12-benchmark-a100.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-13-report.md`
- Create: `docs/challenges/eb-jepa-cifar10/phase-14-cli.md`

**Interfaces:**
- Consumes: le modèle commun, VICReg, loaders, optimisation, configuration et checkpoints.
- Produces: 19 problèmes autonomes allant d'un epoch aux scripts CLI et à la décision scientifique finale.

- [ ] Décrire explicitement le flux de données et les états train/eval/no-grad de chaque boucle.
- [ ] Distinguer validation, test final sans labels et benchmark A100.
- [ ] Détailler unités, warmup de mesure, synchronisation CUDA et agrégation des trois seeds.
- [ ] Donner pour chaque script son invocation exacte et le résultat attendu.

### Task 5: Contrôle pédagogique et cohérence globale

**Files:**
- Modify: tous les fichiers créés dans `docs/challenges/eb-jepa-cifar10/`
- Modify: `docs/superpowers/plans/2026-07-15-eb-jepa-cifar10-self-implementation-roadmap.md`

**Interfaces:**
- Consumes: les 58 énoncés réécrits.
- Produces: une roadmap navigable, cohérente et sans consigne implicite.

- [ ] Compter 59 titres `Exercice` et quatorze rubriques dans chacun.
- [ ] Vérifier que chaque chemin de fichier, symbole consommé et signature est expliqué avant utilisation.
- [ ] Rechercher et corriger les consignes vagues, placeholders et références à des tests automatisés.
- [ ] Contrôler `git diff --check`, les liens locaux et la cohérence des formes 512 vers 2048.
- [ ] Committer uniquement les documents de cette réécriture.
