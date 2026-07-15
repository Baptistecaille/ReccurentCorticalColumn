# Phase 13 — Agréger les seeds et décider

Écris `src/comparison/report.py`. Le rapport utilise trois seeds baseline et trois seeds cortical.

### Exercice 13.1 — `SeedResult`

**Difficulté :** Facile

#### Où travailler

`src/comparison/report.py`.

#### Contexte

Chaque ligne du rapport doit associer architecture, seed, score test, diagnostic, compute et checkpoint.

#### Objectif

Définir une dataclass pour un run complet.

#### Entrées

head_type, training_seed, TestEvaluation, BenchmarkResult.

#### Sorties

Un SeedResult immuable.

#### Comportement attendu

- stocker les quatre champs.
- ne recalculer aucune métrique.

#### Exemple

La ligne `baseline, seed=1000` pointe vers son hash de checkpoint et ses métriques A100.

#### Contraintes

- seed issue de la liste imposée.
- aucune accuracy.
- aucun Tensor direct.

#### Code de départ

```python
@dataclass(frozen=True)
class SeedResult:
    head_type: str
    training_seed: int
    evaluation: TestEvaluation
    benchmark: BenchmarkResult
```

#### Étapes conseillées

1. importe les types.
2. déclare la dataclass.
3. utilise-la comme entrée d'agrégation.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.report import SeedResult; print(SeedResult.__annotations__)"` doit réussir.

#### Définition de terminé

- [ ] quatre champs.
- [ ] types exacts.
- [ ] objet immuable.

#### Indices

1. Une instance représente une seed.
2. Le hash reste dans evaluation.
3. Le nom GPU reste dans benchmark.

---

### Exercice 13.2 — `AggregateResult`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/report.py`.

#### Contexte

Une architecture est résumée sur exactement les seeds 1, 1000 et 10000, sans sélectionner le meilleur run.

#### Objectif

Agréger moyenne et dispersion tout en conservant les résultats bruts.

#### Entrées

Trois SeedResult du même head_type.

#### Sorties

AggregateResult avec score, collapse et compute agrégés.

#### Comportement attendu

- valider un seul head_type.
- valider l'ensemble exact des seeds.
- calculer moyennes et écarts-types.
- conserver le tuple des trois runs.

#### Exemple

Les losses [1.0,1.2,0.8] donnent une moyenne 1.0 ; aucune seed n'est supprimée.

#### Contraintes

- exactement trois seeds.
- ddof documenté.
- aucune sélection du meilleur.

#### Code de départ

```python
@dataclass(frozen=True)
class AggregateResult:
    head_type: str
    runs: tuple[SeedResult, SeedResult, SeedResult]
    mean_test_score: float
    std_test_score: float
    # Ajoute les agrégats collapse et compute nécessaires.


def aggregate_results(results: Sequence[SeedResult]) -> AggregateResult:
    ...
```

#### Étapes conseillées

1. valide type et seeds.
2. extrais les métriques.
3. calcule mean/std.
4. retourne.

#### Vérification manuelle

Avec trois faux runs, vérifie la moyenne et le refus d'une seed dupliquée.

#### Définition de terminé

- [ ] seeds exactes.
- [ ] résultats bruts conservés.
- [ ] agrégats corrects.

#### Indices

1. Compare le set à `{1,1000,10000}`.
2. Agrège les architectures séparément.
3. Choisis un ddof puis garde-le partout.

---

### Exercice 13.3 — `ComparisonDecision`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/report.py`.

#### Contexte

La réussite exige simultanément qualité comparable, absence d'effondrement et compute inférieur.

#### Objectif

Encoder les trois critères et leurs écarts.

#### Entrées

Agrégats baseline/cortical et tolérance relative 0.02.

#### Sorties

ComparisonDecision avec quatre booléens et deux pourcentages.

#### Comportement attendu

- calculer l'écart relatif de loss.
- évaluer le critère d'effondrement.
- calculer la réduction de compute.
- définir success comme conjonction.

#### Exemple

1% de loss en plus et 30% de compute en moins sans collapse réussit ; 3% de loss en plus échoue.

#### Contraintes

- une loss plus basse est meilleure.
- tolérance relative au baseline.
- métrique compute nommée.

#### Code de départ

```python
@dataclass(frozen=True)
class ComparisonDecision:
    quality_ok: bool
    collapse_ok: bool
    compute_ok: bool
    success: bool
    relative_score_gap: float
    compute_reduction: float


def decide_comparison(baseline: AggregateResult, cortical: AggregateResult,
                      score_tolerance: float = 0.02) -> ComparisonDecision:
    ...
```

#### Étapes conseillées

1. écris les deux formules de pourcentage.
2. calcule trois critères.
3. compose success.

#### Vérification manuelle

Contrôle trois cas : succès, score hors tolérance, compute non réduit.

#### Définition de terminé

- [ ] conjonction correcte.
- [ ] sens de la loss.
- [ ] pourcentages explicites.

#### Indices

1. Gap=(cortical-baseline)/baseline.
2. Réduction=1-cortical/baseline.
3. Une condition ne compense jamais une autre.

---

### Exercice 13.4 — `write_report`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/report.py`.

#### Contexte

Markdown sert au lecteur et JSON à l'audit. Les deux fichiers doivent présenter les mêmes six runs.

#### Objectif

Écrire les résultats bruts, agrégats, décision et provenance.

#### Entrées

Deux chemins, deux AggregateResult et ComparisonDecision.

#### Sorties

Un `.md` et un `.json` cohérents.

#### Comportement attendu

- table des six runs.
- table des deux agrégats.
- détail des trois critères.
- GPU, configs et hashes.
- JSON avec allow_nan=False.

#### Exemple

Le lecteur voit l'écart de score, le compute économisé et chaque seed avant la conclusion.

#### Contraintes

- aucun run omis.
- unités explicites.
- aucune affirmation non mesurée.

#### Code de départ

```python
def write_report(markdown_path: str | Path, json_path: str | Path,
                 baseline: AggregateResult, cortical: AggregateResult,
                 decision: ComparisonDecision) -> None:
    ...
```

#### Étapes conseillées

1. compose un payload.
2. écris JSON.
3. rends les tables Markdown.
4. ajoute la conclusion.

#### Vérification manuelle

`python -m json.tool report.json` réussit et le Markdown contient six lignes, `A100` et `VICReg`.

#### Définition de terminé

- [ ] six runs.
- [ ] hashes et unités.
- [ ] JSON valide.
- [ ] conclusion fidèle.

#### Indices

1. Pars des dataclasses.
2. Utilise le même payload pour les deux formats.
3. Affiche les valeurs brutes avant la décision.

