# Phase 8 — Charger et valider la configuration

Écris `src/comparison/config.py` et deux fichiers YAML. `DictConfig` est l'objet dictionnaire d'OmegaConf : il permet `cfg.model.head_type` tout en conservant une structure sérialisable.

### Exercice 8.1 — `load_config`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/config.py`. Source : chargement OmegaConf de l'exemple officiel.

#### Contexte

Un `DictConfig` remplace une longue liste d'arguments CLI. Les overrides pointés, par exemple `optimization.epochs=2`, permettent un essai court sans modifier le YAML.

#### Objectif

Charger un YAML, fusionner les overrides CLI et retourner la configuration validée.

#### Entrées

`path: str|Path`; `overrides: list[str]` au format clé.pointée=valeur.

#### Sorties

`DictConfig` structuré.

#### Comportement attendu

- charger avec OmegaConf.load.
- convertir les overrides avec OmegaConf.from_dotlist.
- fusionner base puis overrides.
- appeler validate_config.
- retourner cfg.

#### Exemple

`load_config('configs/baseline.yaml',['optimization.epochs=2'])` conserve toutes les clés mais remplace epochs par l'entier 2.

#### Contraintes

- override prioritaire.
- types parsés par OmegaConf.
- fichier absent signalé clairement.
- aucune mutation du YAML.

#### Code de départ

```python
def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    ...
```

#### Étapes conseillées

1. charge le document.
2. crée une config vide si overrides None.
3. fusionne dans le bon ordre.
4. valide puis retourne.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.config import load_config; c=load_config('configs/baseline.yaml',['optimization.epochs=2']); print(type(c).__name__,c.optimization.epochs)"` affiche `DictConfig 2`.

#### Définition de terminé

- [ ] retour DictConfig.
- [ ] override appliqué.
- [ ] validation appelée.

#### Indices

1. OmegaConf.merge donne priorité aux arguments de droite.
2. from_dotlist comprend les points.
3. Ne convertis pas en dict Python avant validation.

---

### Exercice 8.2 — `validate_config`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/config.py`, appelée par load_config.

#### Contexte

Une mauvaise dimension ou un batch trop petit peut échouer après plusieurs minutes. Cette fonction centralise les invariants du protocole avant tout accès au GPU.

#### Objectif

Énumérer et vérifier chaque invariant structurel, data, loss et benchmark.

#### Entrées

`cfg: DictConfig`.

#### Sorties

Aucun retour utile si valide ; `ValueError` explicite sinon.

#### Comportement attendu

- head_type dans baseline/cortical.
- feature_dim=512 et output_dim=2048.
- tailles 45k/5k/10k et batch>1.
- coefficients/profondeurs valides.
- seeds exactement [1,1000,10000].
- precision bfloat16 et device A100 requis seulement au benchmark.

#### Exemple

Une config cortical avec `dim_in=256` échoue en mentionnant `cortical.dim_in`, attendu 512 et reçu 256.

#### Contraintes

- une erreur par invariant avec chemin de clé.
- ne corrige jamais silencieusement.
- ne charge ni données ni modèle.
- type inconnu rejeté.

#### Code de départ

```python
def validate_config(cfg: DictConfig) -> None:
    ...
```

#### Étapes conseillées

1. liste les invariants sur papier.
2. vérifie présence puis valeur.
3. écris des messages chemin/attendu/reçu.
4. termine sans return nécessaire.

#### Vérification manuelle

`PYTHONPATH=src python -c "from omegaconf import OmegaConf; from comparison.config import validate_config; validate_config(OmegaConf.create({'model':{'head_type':'bad'}}))"` doit échouer en mentionnant `model.head_type`.

#### Définition de terminé

- [ ] messages actionnables.
- [ ] dimensions contrôlées.
- [ ] aucune correction implicite.
- [ ] config valide acceptée.

#### Indices

1. Commence par les clés indispensables.
2. Une petite fonction require(condition,message) évite les répétitions.
3. Les contraintes globales sont listées dans l'index de roadmap.

---

### Exercice 8.3 — `YAML baseline et cortical`

**Difficulté :** Moyen

#### Où travailler

Crée `configs/baseline.yaml` et `configs/cortical.yaml`. Source : `external/eb_jepa/examples/image_jepa/cfgs/default.yaml`, adaptée au protocole fixé.

#### Contexte

Les deux expériences doivent différer uniquement par `model.head_type` et la section de paramètres réellement propre à la tête. Les valeurs communes ne doivent pas dériver.

#### Objectif

Écrire deux configurations complètes, lisibles et validables.

#### Entrées

Aucune entrée runtime ; documents YAML.

#### Sorties

Deux DictConfig ayant les mêmes data/loss/optimization/experiment et des têtes différentes.

#### Comportement attendu

- copier les valeurs officielles utiles.
- ajouter les tailles CIFAR et dimensions.
- définir baseline puis dupliquer les valeurs communes.
- changer seulement head_type et ajouter cortical.

#### Exemple

Une comparaison des sections `data`, `loss`, `optimization`, `experiment` donne une égalité exacte entre les deux fichiers.

#### Contraintes

- batch 256, epochs 300, LR .3, warmup 10.
- VICReg 1/1/80.
- seeds 1/1000/10000.
- aucune clé linear_probe ou wandb.

#### Code de départ

```python
# configs/baseline.yaml
data:
  dataset: cifar10
  train_size: 45000
  validation_size: 5000
  test_size: 10000
model:
  head_type: baseline
  feature_dim: 512
  output_dim: 2048
# Complète toutes les sections imposées.
```

#### Étapes conseillées

1. reporte le protocole commun.
2. écris baseline complet.
3. copie-le en cortical.
4. modifie uniquement les clés de tête.
5. charge les deux.

#### Vérification manuelle

`PYTHONPATH=src python -c "from omegaconf import OmegaConf as O; a=O.load('configs/baseline.yaml'); b=O.load('configs/cortical.yaml'); print(a.data==b.data,a.optimization==b.optimization,a.model.head_type,b.model.head_type)"` affiche `True True baseline cortical`.

#### Définition de terminé

- [ ] deux YAML chargeables.
- [ ] sections communes égales.
- [ ] dimensions cohérentes.
- [ ] aucune clé interdite.

#### Indices

1. YAML utilise deux espaces d'indentation.
2. Les nombres scientifiques sont acceptés.
3. La section cortical peut exister dans les deux fichiers si cela simplifie l'égalité.

