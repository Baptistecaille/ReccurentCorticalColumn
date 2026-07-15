# Phase 11 — Calculer le score test sans labels

Écris `src/comparison/evaluate.py`. Commence uniquement après gel de la configuration et choix du meilleur checkpoint.

### Exercice 11.1 — `PairEvaluation`

**Difficulté :** Facile

#### Où travailler

`src/comparison/evaluate.py`.

#### Contexte

Chaque paire déterministe produit une loss et ses composantes. Un type nommé évite de confondre ces valeurs.

#### Objectif

Définir la dataclass d'un score pour une seed de paires.

#### Entrées

Valeurs float calculées sur tout le loader.

#### Sorties

Objet PairEvaluation sérialisable.

#### Comportement attendu

- champs total, invariance, variance, covariance.
- champs mean_std, min_std, collapsed_fraction.
- champ pair_seed.

#### Exemple

Une évaluation pour pair_seed 17 garde toutes les composantes nécessaires à l'audit.

#### Contraintes

- frozen.
- aucun Tensor.
- aucune accuracy.

#### Code de départ

```python
@dataclass(frozen=True)
class PairEvaluation:
    pair_seed: int
    total: float
    invariance: float
    variance: float
    covariance: float
    mean_std: float
    min_std: float
    collapsed_fraction: float
```

#### Étapes conseillées

1. déclare la dataclass.
2. reprends les noms dans evaluate_pair.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.evaluate import PairEvaluation; print(PairEvaluation.__annotations__)"` affiche huit champs.

#### Définition de terminé

- [ ] huit champs.
- [ ] aucune accuracy.
- [ ] types simples.

#### Indices

1. Le score principal est total.
2. Les composantes expliquent le total.
3. pair_seed concerne les vues.

---

### Exercice 11.2 — `evaluate_pair`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/evaluate.py`.

#### Contexte

Une paire test doit être évaluée comme la validation, mais toutes les composantes et diagnostics sont rapportés.

#### Objectif

Parcourir un DeterministicPairDataset une fois en eval/no_grad.

#### Entrées

Model, loader, loss, pair_seed, device, BF16.

#### Sorties

PairEvaluation agrégée.

#### Comportement attendu

- eval/no_grad/autocast.
- deux forwards.
- calculer composantes sans labels.
- agréger par taille batch.
- diagnostiquer représentations.

#### Exemple

Un loader de 10k images donne un seul objet ; changer pair_seed exige de reconstruire le dataset.

#### Contraintes

- pas d'optimizer.
- pas de shuffle.
- aucun label.
- agrégation pondérée.

#### Code de départ

```python
def evaluate_pair(model: ImageSSL, loader: DataLoader, loss_fn: VICRegLoss,
                  pair_seed: int, device: torch.device, use_bf16: bool) -> PairEvaluation:
    ...
```

#### Étapes conseillées

1. reprends validation.
2. expose composantes.
3. agrège.
4. construis dataclass.

#### Vérification manuelle

Sur un loader miniature fixe, deux appels donnent les mêmes floats.

#### Définition de terminé

- [ ] déterministe.
- [ ] paramètres inchangés.
- [ ] composantes finies.

#### Indices

1. Ne moyenne pas des moyennes sans poids.
2. Les deux vues contribuent au diagnostic.
3. Conserve pair_seed.

---

### Exercice 11.3 — `TestEvaluation`

**Difficulté :** Facile

#### Où travailler

`src/comparison/evaluate.py`.

#### Contexte

Le score final moyenne cinq créations de paires afin de réduire la dépendance à un tirage particulier.

#### Objectif

Regrouper cinq PairEvaluation et leur moyenne/écart-type.

#### Entrées

Tuple de cinq résultats et hash checkpoint.

#### Sorties

Objet TestEvaluation.

#### Comportement attendu

- stocker pairs.
- calculer mean/std total.
- conserver checkpoint_sha256.

#### Exemple

Cinq pair seeds donnent cinq losses ; le rapport ne choisit jamais la meilleure.

#### Contraintes

- exactement cinq.
- résultats bruts conservés.
- ddof documenté.

#### Code de départ

```python
@dataclass(frozen=True)
class TestEvaluation:
    pairs: tuple[PairEvaluation, ...]
    mean_total: float
    std_total: float
    checkpoint_sha256: str
```

#### Étapes conseillées

1. déclare le type.
2. utilise-le dans evaluate_test.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.evaluate import TestEvaluation; print(TestEvaluation.__annotations__)"` doit réussir.

#### Définition de terminé

- [ ] pairs conservés.
- [ ] mean/std.
- [ ] hash présent.

#### Indices

1. Un tuple protège la collection.
2. Le hash relie score et poids.
3. Ne confonds pas seeds.

---

### Exercice 11.4 — `evaluate_test`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/evaluate.py`.

#### Contexte

Cette fonction est l'unique porte d'accès au test. Elle recharge le meilleur checkpoint puis crée cinq datasets avec seeds fixées.

#### Objectif

Évaluer exactement cinq paires sans ajuster le modèle.

#### Entrées

Cfg gelée, checkpoint, cinq seeds.

#### Sorties

TestEvaluation.

#### Comportement attendu

- valider seeds distinctes.
- construire modèle et charger strictement.
- reconstruire loader pour chaque seed.
- appeler evaluate_pair.
- agréger et hasher.

#### Exemple

Les mêmes entrées reproduisent le même JSON ; aucun résultat ne modifie les poids.

#### Contraintes

- aucun backward.
- aucun réglage après score.
- exactement cinq seeds.
- best checkpoint.

#### Code de départ

```python
def evaluate_test(cfg: DictConfig, checkpoint_path: str | Path,
                  pair_seeds: tuple[int, int, int, int, int]) -> TestEvaluation:
    ...
```

#### Étapes conseillées

1. valide seeds.
2. charge modèle.
3. boucle.
4. agrège.
5. calcule SHA-256.

#### Vérification manuelle

Exécute deux fois sur le même petit checkpoint et compare les objets.

#### Définition de terminé

- [ ] cinq paires.
- [ ] hash exact.
- [ ] reproductible.
- [ ] aucune mutation.

#### Indices

1. Le loader dépend de pair_seed.
2. SHA-256 porte sur les octets.
3. run n'appelle jamais cette fonction.

---

### Exercice 11.5 — `write_evaluation_json`

**Difficulté :** Facile

#### Où travailler

`src/comparison/evaluate.py`.

#### Contexte

Le score doit être archivé avec configuration et checkpoint pour être auditable.

#### Objectif

Sérialiser evaluation, config et métadonnées.

#### Entrées

Path, evaluation, cfg, seed entraînement.

#### Sorties

JSON UTF-8 stable.

#### Comportement attendu

- dataclasses.asdict.
- OmegaConf.to_container.
- ajouter architecture/seed.
- indent et sort_keys.
- allow_nan=False.

#### Exemple

Le JSON expose cinq pairs et le hash sans charger PyTorch.

#### Contraintes

- aucun Tensor.
- NaN interdit.
- parent créé.

#### Code de départ

```python
def write_evaluation_json(path: str | Path, evaluation: TestEvaluation,
                          cfg: DictConfig, training_seed: int) -> None:
    ...
```

#### Étapes conseillées

1. convertis les objets.
2. compose payload.
3. écris JSON.

#### Vérification manuelle

`python -m json.tool result.json` doit réussir.

#### Définition de terminé

- [ ] JSON valide.
- [ ] cinq pairs.
- [ ] config/hash.

#### Indices

1. asdict aide.
2. to_container résout cfg.
3. allow_nan=False détecte une métrique invalide.

