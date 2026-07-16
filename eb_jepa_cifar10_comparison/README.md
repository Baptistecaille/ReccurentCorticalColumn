# EB-JEPA CIFAR-10 Predictor Comparison

Ce dossier accueillera l’implémentation personnelle de la comparaison entre le
projecteur MLP de l’exemple Image JEPA et `FixedTreePredictor`.

La référence officielle intacte se trouve dans `../external/eb_jepa` au commit
`966e61e9285b3a876f49b9774e9720d9a99a7925`.

L’implémentation doit suivre la roadmap :
`../docs/superpowers/plans/2026-07-15-eb-jepa-cifar10-self-implementation-roadmap.md`.

Aucun fichier de code officiel ne doit être modifié dans `external/eb_jepa`.

## Workflow interactif

Depuis un checkout frais, placez-vous dans `eb_jepa_cifar10_comparison`, puis
installez les dépendances runtime et notebook dans l’environnement du projet :

```bash
uv sync --extra notebook
uv run jupyter lab scripts/workflow.ipynb
```

Dans JupyterLab, sélectionnez le kernel Python de l’environnement créé par
`uv` pour ce projet. Exécutez d’abord une fois la cellule d’imports partagés.
Pour utiliser ensuite une section, modifiez uniquement sa cellule de
paramètres, puis exécutez sa cellule d’exécution. Chaque section est ainsi
exécutable indépendamment des autres après les imports partagés.

### Paramètres éditables

| Section | Paramètres | Rôle |
|---|---|---|
| Train | `TRAIN_CONFIG`, `TRAIN_OVERRIDES`, `TRAIN_SEED`, `TRAIN_OUTPUT_DIR`, `TRAIN_RESUME_FROM` | Configuration YAML, overrides OmegaConf, graine d’entraînement, dossier de run et checkpoint optionnel de reprise |
| Evaluate | `EVALUATE_CONFIG`, `EVALUATE_OVERRIDES`, `EVALUATE_CHECKPOINT`, `EVALUATE_PAIR_SEEDS`, `EVALUATE_OUTPUT` | Configuration, overrides, checkpoint local, exactement cinq graines de paires distinctes et artefact partagé |
| Benchmark | `BENCHMARK_CONFIG`, `BENCHMARK_OVERRIDES`, `BENCHMARK_CHECKPOINT`, `BENCHMARK_OUTPUT` | Configuration, overrides, même checkpoint local et même artefact partagé |
| Report | `REPORT_RESULTS`, `REPORT_MARKDOWN_OUTPUT`, `REPORT_JSON_OUTPUT`, `REPORT_SCORE_TOLERANCE` | Exactement six artefacts complets, sorties Markdown/JSON distinctes et tolérance relative (défaut `0.02`, soit 2 %) |

Un override est une chaîne telle que `"optimization.epochs=10"` dans la liste
`*_OVERRIDES` concernée. `TRAIN_RESUME_FROM=None` démarre un nouveau run ; une
autre valeur doit pointer vers le checkpoint de reprise. Les checkpoints
d’Evaluate et Benchmark doivent être des fichiers locaux fiables : leur
chargement emploie `torch.load(weights_only=False)`, qui peut exécuter du code
contenu dans un pickle non fiable.

Le flux normal est **Train → Evaluate → Benchmark → Report** :

1. **Train** produit le meilleur checkpoint pour une architecture et une
   graine d’entraînement.
2. **Evaluate** évalue ce checkpoint avec les cinq graines de paires et écrit
   ses résultats dans le `result.json` de l’exécution.
3. **Benchmark** mesure le même checkpoint et complète ce même `result.json`.
   Les paramètres `EVALUATE_OUTPUT` et `BENCHMARK_OUTPUT` doivent donc désigner
   le même chemin pour une architecture et une graine données. Pour un run,
   exécutez Evaluate puis Benchmark séquentiellement, sans exécutions
   concurrentes qui écriraient simultanément dans cet artefact.
4. **Report** consomme exactement six `result.json` complets : trois pour
   Baseline et trois pour Cortical, avec les graines d’entraînement `1`, `1000`
   et `10000` pour chaque architecture. Les entrées attendues sont :

   - `runs/baseline/1/result.json`
   - `runs/baseline/1000/result.json`
   - `runs/baseline/10000/result.json`
   - `runs/cortical/1/result.json`
   - `runs/cortical/1000/result.json`
   - `runs/cortical/10000/result.json`

Le benchmark applique strictement le protocole **NVIDIA A100, bfloat16 et
batch de 256**. Il n’existe aucun fallback CPU, autre GPU ou taille de batch en
cas d’indisponibilité ou d’erreur de mémoire : exécutez la section Benchmark sur
une A100 conforme au protocole.
