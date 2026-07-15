# Phase 10 — Entraîner avec une boucle commune

Écris `src/comparison/train.py`. Aucune fonction de cette phase ne teste `head_type` : le polymorphisme de `ImageSSL` suffit.

### Exercice 10.1 — `train_epoch`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/train.py`. Adapte le corps d'entraînement de l'exemple Image officiel.

#### Contexte

Chaque batch contient deux vues. Elles passent séparément dans le même modèle ; VICReg compare leurs projections. Le scheduler avance à chaque mise à jour, pas à chaque epoch.

#### Objectif

Exécuter un epoch et retourner les moyennes de loss sans conserver les graphes.

#### Entrées

Model, DataLoader, LARS, scheduler, VICRegLoss, device et mode BF16.

#### Sorties

Métriques moyennes de l'epoch et nombre de batches.

#### Comportement attendu

- model.train.
- déplacer deux vues sur device.
- zero_grad.
- deux forwards et extraction projections.
- loss, backward, optimizer.step, scheduler.step.
- accumuler des floats pondérés par batch.

#### Exemple

Un batch `(x1,x2)` produit `(_,z1)` et `(_,z2)` ; seule `loss_fn(z1,z2)` est rétropropagée.

#### Contraintes

- aucun label.
- même autocast pour les deux têtes.
- scheduler après optimizer.
- pas de condition architecture.
- loss non finie refusée.

#### Code de départ

```python
def train_epoch(model: ImageSSL, train_loader: DataLoader, optimizer: LARS,
                scheduler: WarmupCosineScheduler, loss_fn: VICRegLoss,
                device: torch.device, use_bf16: bool) -> EpochMetrics:
    ...
```

#### Étapes conseillées

1. active train.
2. écris le traitement d'un batch.
3. ajoute backward et mises à jour.
4. agrège après item.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.train import train_epoch; print(train_epoch.__annotations__)"` doit importer la fonction ; un essai sur deux petits batches doit retourner une loss finie.

#### Définition de terminé

- [ ] ordre des opérations correct.
- [ ] deux vues.
- [ ] scheduler par batch.
- [ ] moyenne correcte.

#### Indices

1. ImageSSL retourne un tuple.
2. autocast peut recevoir enabled=use_bf16.
3. Pondère par nombre d'exemples si le dernier batch varie.

---

### Exercice 10.2 — `evaluate_validation`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/train.py`, après train_epoch.

#### Contexte

La validation choisit le meilleur checkpoint sans mettre à jour paramètres, BatchNorm ou gradients. Elle utilise des paires déterministes.

#### Objectif

Calculer VICReg et diagnostics sur tout le loader en mode évaluation.

#### Entrées

Model, loader validation, loss, device, BF16.

#### Sorties

Métriques moyennes et diagnostic agrégé.

#### Comportement attendu

- model.eval et torch.no_grad.
- deux forwards sous autocast.
- aucun backward/step.
- agréger loss et représentations nécessaires.

#### Exemple

Deux appels consécutifs sur le même loader et checkpoint donnent les mêmes métriques.

#### Contraintes

- optimizer absent de la signature.
- aucun gradient.
- paires déterministes.
- aucun test split ici.

#### Code de départ

```python
def evaluate_validation(model: ImageSSL, loader: DataLoader, loss_fn: VICRegLoss,
                        device: torch.device, use_bf16: bool) -> ValidationMetrics:
    ...
```

#### Étapes conseillées

1. passe en eval.
2. ouvre no_grad.
3. réutilise le calcul à deux vues.
4. agrège puis retourne.

#### Vérification manuelle

Exécute deux fois sur un loader miniature fixe et compare les dataclasses ; elles doivent être égales à l'arrondi près.

#### Définition de terminé

- [ ] paramètres inchangés.
- [ ] résultat répétable.
- [ ] loss finie.
- [ ] model.eval utilisé.

#### Indices

1. eval fige BatchNorm.
2. no_grad empêche backward.
3. Ne réutilise pas train_epoch avec optimizer=None.

---

### Exercice 10.3 — `RunResult`

**Difficulté :** Facile

#### Où travailler

`src/comparison/train.py`, près de run.

#### Contexte

Le script CLI a besoin d'un résultat structuré : meilleur checkpoint, meilleure validation, epoch et historique minimal.

#### Objectif

Créer une dataclass immuable décrivant les sorties observables d'un run.

#### Entrées

Path et nombres calculés par run.

#### Sorties

Objet sérialisable ne contenant ni Tensor GPU ni module.

#### Comportement attendu

- définir les quatre champs.
- ne lancer aucune opération dans la dataclass.

#### Exemple

`RunResult(Path('best.pt'),1.2,87,300)` décrit le meilleur point d'un run de 300 epochs.

#### Contraintes

- frozen.
- pas d'objet CUDA.
- types explicites.
- chemin obligatoire.

#### Code de départ

```python
@dataclass(frozen=True)
class RunResult:
    best_checkpoint: Path
    best_validation: float
    best_epoch: int
    final_epoch: int
```

#### Étapes conseillées

1. importe dataclass et Path.
2. déclare les champs.
3. utilise ce type dans run.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.train import RunResult; from pathlib import Path; print(RunResult(Path('x'),1.,2,3))"` doit réussir.

#### Définition de terminé

- [ ] quatre champs présents.
- [ ] objet immuable.
- [ ] valeurs sérialisables.

#### Indices

1. Une dataclass suffit.
2. Le checkpoint final n'est pas nécessairement le meilleur.
3. L'epoch est un entier, la loss un float.

---

### Exercice 10.4 — `run`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/train.py`. Adapte l'ordre du main officiel.

#### Contexte

Cette orchestration relie config, device, seed, loaders, modèle, loss, optimizer, scheduler et checkpoints. L'ordre est important pour la reproductibilité.

#### Objectif

Effectuer un run complet ou reprendre un checkpoint, sélectionner sur validation et retourner RunResult.

#### Entrées

Cfg validée, seed, head choisi par cfg, chemin de sortie, checkpoint optionnel.

#### Sorties

RunResult et fichiers best/periodic.

#### Comportement attendu

- setup device puis seed.
- construire transforms/loaders/model/loss/optim/scheduler.
- reprendre si demandé.
- boucler train puis validation.
- sauver best et périodiques.
- ne jamais évaluer test.

#### Exemple

Avec `epochs=2`, le run effectue deux validations et `best_checkpoint` pointe vers la plus faible loss validation.

#### Contraintes

- aucun accès au test loader.
- aucune branche head_type.
- OOM propagée.
- checkpoint_every respecté.

#### Code de départ

```python
def run(cfg: DictConfig, *, seed: int, output_dir: str | Path,
        resume_from: str | Path | None = None) -> RunResult:
    ...
```

#### Étapes conseillées

1. écris la construction dans l'ordre.
2. calcule les steps scheduler.
3. gère reprise.
4. boucle et compare validation.
5. retourne.

#### Vérification manuelle

Lance avec `optimization.epochs=1 data.num_workers=0` ; vérifie un best checkpoint et aucune évaluation test.

#### Définition de terminé

- [ ] run court aboutit.
- [ ] best sauvegardé.
- [ ] reprise disponible.
- [ ] test jamais lu.

#### Indices

1. Construis tout après setup_seed.
2. Une loss plus basse est meilleure.
3. La phase 11 seule utilise le test.

