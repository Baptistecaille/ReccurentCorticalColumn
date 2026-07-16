# EB-JEPA CIFAR-10 Predictor Comparison

Ce dossier accueillera l’implémentation personnelle de la comparaison entre le
projecteur MLP de l’exemple Image JEPA et `FixedTreePredictor`.

La référence officielle intacte se trouve dans `../external/eb_jepa` au commit
`966e61e9285b3a876f49b9774e9720d9a99a7925`.

L’implémentation doit suivre la roadmap :
`../docs/superpowers/plans/2026-07-15-eb-jepa-cifar10-self-implementation-roadmap.md`.

Aucun fichier de code officiel ne doit être modifié dans `external/eb_jepa`.

## Workflow interactif

Depuis le dossier `eb_jepa_cifar10_comparison`, ouvrez
`scripts/workflow.ipynb` dans Jupyter. Exécutez d’abord une fois la cellule
d’imports partagés. Pour utiliser ensuite une section, modifiez uniquement sa
cellule de paramètres, puis exécutez sa cellule d’exécution. Chaque section est
ainsi exécutable indépendamment des autres après les imports partagés.

Le flux normal est **Train → Evaluate → Benchmark → Report** :

1. **Train** produit le meilleur checkpoint pour une architecture et une
   graine d’entraînement.
2. **Evaluate** évalue ce checkpoint avec les cinq graines de paires et écrit
   ses résultats dans le `result.json` de l’exécution.
3. **Benchmark** mesure le même checkpoint et complète ce même `result.json`.
   Les paramètres `EVALUATE_OUTPUT` et `BENCHMARK_OUTPUT` doivent donc désigner
   le même chemin pour une architecture et une graine données.
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
