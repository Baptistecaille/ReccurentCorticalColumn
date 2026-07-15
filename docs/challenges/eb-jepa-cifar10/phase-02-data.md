# Phase 2 — Adapter les datasets et les splits

Cette phase écrit `src/comparison/data.py`. Les labels CIFAR-10 sont volontairement retirés de toutes les sorties.

### Exercice 2.1 — `PairedViewDataset`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/data.py`. Adapte le principe de `ImageDataset` dans `external/eb_jepa/examples/image_jepa/dataset.py`.

#### Contexte

VICReg compare deux vues aléatoires de la même image. Le dataset torchvision retourne `(image,label)`, mais le label est interdit dans ce projet.

#### Objectif

Créer une enveloppe qui ignore le label et appelle deux fois le même transform.

#### Entrées

`dataset`, dataset indexable retournant image ou `(image,label)` ; `transform`, callable stochastique.

#### Sorties

Tuple `(view1, view2)` ; chaque vue Tensor `(3,32,32)`.

#### Comportement attendu

- déléguer la longueur au dataset source.
- extraire uniquement l'image à l'index.
- appeler deux fois transform séparément.

#### Exemple

Pour une image d'indice 7, les deux vues ont la même source mais des crops et couleurs potentiellement différents.

#### Contraintes

- ne retourne jamais le label.
- ne réutilise pas le même résultat Tensor.
- ne change pas l'ordre des indices.

#### Code de départ

```python
class PairedViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...
```

#### Étapes conseillées

1. stocke dataset et transform.
2. récupère l'élément source.
3. isole l'image puis appelle transform deux fois.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.data import PairedViewDataset; d=[('x',9)]; p=PairedViewDataset(d,lambda x:x+'v'); print(p[0])"` affiche `('xv', 'xv')` et jamais `9`.

#### Définition de terminé

- [ ] longueur déléguée.
- [ ] label absent.
- [ ] deux appels au transform par index.

#### Indices

1. Un tuple torchvision contient l'image en position 0.
2. Deux vues ne signifient pas deux images source.
3. Ne mets pas le transform dans le dataset CIFAR sous-jacent.

---

### Exercice 2.2 — `SplitIndices`

**Difficulté :** Facile

#### Où travailler

`src/comparison/data.py`, près des imports.

#### Contexte

Les indices du train et de la validation doivent être enregistrés ensemble afin d'éviter une inversion ou un recouvrement.

#### Objectif

Créer une dataclass immuable contenant deux listes d'indices.

#### Entrées

`train` et `validation`, listes d'entiers.

#### Sorties

Instance `SplitIndices` lisible et non réassignable.

#### Comportement attendu

- déclarer frozen=True.
- annoter les deux champs.
- ne calculer aucun split dans la classe.

#### Exemple

`SplitIndices(train=[0,2], validation=[1])` décrit trois positions sans stocker d'images.

#### Contraintes

- pas de champ test : CIFAR fournit déjà son test.
- pas de logique dans la dataclass.
- types explicites.

#### Code de départ

```python
@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]
```

#### Étapes conseillées

1. importe dataclass.
2. écris le décorateur.
3. ajoute les champs.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.data import SplitIndices; print(SplitIndices([0],[1]))"` affiche les deux champs.

#### Définition de terminé

- [ ] classe immuable.
- [ ] deux champs seulement.
- [ ] annotations list[int].

#### Indices

1. Une dataclass génère le constructeur.
2. frozen empêche la réassignation des champs.
3. Les listes internes ne deviennent pas des tuples automatiquement.

---

### Exercice 2.3 — `build_split_indices`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/data.py`, après `SplitIndices`.

#### Contexte

Le même découpage 45k/5k doit servir aux deux têtes et aux trois seeds d'architecture ; seul `split_seed` détermine l'ordre.

#### Objectif

Produire une permutation locale de 50 000 indices et la découper.

#### Entrées

`dataset_size=50000`, `validation_size=5000`, `split_seed`.

#### Sorties

`SplitIndices` avec 45 000 train et 5 000 validation sans chevauchement.

#### Comportement attendu

- valider les tailles.
- créer un torch.Generator local.
- seed puis randperm.
- répartir validation et train de façon documentée.

#### Exemple

Deux appels avec seed 42 retournent les mêmes listes ; seed 43 produit une autre permutation.

#### Contraintes

- ne modifie pas la seed globale.
- chaque indice apparaît exactement une fois.
- refuse validation_size hors de 1..dataset_size-1.

#### Code de départ

```python
def build_split_indices(dataset_size: int, validation_size: int, split_seed: int) -> SplitIndices:
    ...
```

#### Étapes conseillées

1. valide les bornes.
2. initialise un Generator.
3. convertis randperm en liste.
4. découpe une seule fois.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.data import build_split_indices as f; s=f(10,2,7); print(len(s.train),len(s.validation),len(set(s.train+s.validation)))"` affiche `8 2 10`.

#### Définition de terminé

- [ ] déterministe.
- [ ] aucun chevauchement.
- [ ] couverture complète.
- [ ] seed globale intacte.

#### Indices

1. `torch.randperm` accepte generator.
2. Une permutation garantit unicité et couverture.
3. Choisis et documente si validation prend le début ou la fin.

---

### Exercice 2.4 — `DeterministicPairDataset`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/data.py`.

#### Contexte

Validation et test doivent comparer des runs sur exactement les mêmes vues. Un résultat stochastique à chaque lecture ajouterait du bruit au score.

#### Objectif

Créer des paires reproductibles par `(pair_seed,index)` tout en restaurant l'état aléatoire extérieur.

#### Entrées

`dataset`, `transform`, `pair_seed`, `index`.

#### Sorties

Deux vues déterministes de la même image, sans label.

#### Comportement attendu

- dériver deux seeds distinctes de pair_seed et index.
- exécuter chaque transformation dans un contexte aléatoire isolé.
- restaurer les états Python, NumPy et PyTorch après lecture.

#### Exemple

Lire deux fois l'index 12 avec pair_seed 100 produit deux tenseurs identiques à chaque lecture ; changer pair_seed change les vues.

#### Contraintes

- aucune perturbation du RNG global.
- view1 et view2 utilisent des seeds différentes.
- compatible avec workers DataLoader.

#### Code de départ

```python
class DeterministicPairDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable, pair_seed: int) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...
```

#### Étapes conseillées

1. écris un helper de contexte seed.
2. dérive une seed par vue.
3. transforme la même image deux fois.
4. restaure les états même en cas d'erreur.

#### Vérification manuelle

Dans un interpréteur, compare `torch.equal(d[0][0], d[0][0])` ; le résultat doit être `True`.

#### Définition de terminé

- [ ] lecture répétable.
- [ ] labels absents.
- [ ] RNG extérieur inchangé.
- [ ] deux vues distinctement seedées.

#### Indices

1. `torch.random.fork_rng` aide pour PyTorch.
2. Sauvegarde/restaure aussi random et NumPy si la pipeline les utilise.
3. Combine seed et index sans utiliser hash(), qui peut varier.

---

### Exercice 2.5 — `DataLoaders`

**Difficulté :** Facile

#### Où travailler

`src/comparison/data.py`, près des fabriques.

#### Contexte

Retourner trois loaders séparément rend facile de les inverser. Une dataclass nomme leur rôle.

#### Objectif

Regrouper train, validation et test dans une dataclass immuable.

#### Entrées

Trois instances `DataLoader`.

#### Sorties

Objet avec attributs `.train`, `.validation`, `.test`.

#### Comportement attendu

- déclarer trois champs.
- annoter DataLoader.
- ne construire aucun loader ici.

#### Exemple

`loaders.validation` est explicite et ne peut pas être confondu avec `loaders.test`.

#### Contraintes

- trois champs exactement.
- frozen=True.
- aucune valeur par défaut.

#### Code de départ

```python
@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
```

#### Étapes conseillées

1. importe DataLoader.
2. ajoute le décorateur.
3. déclare les champs.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.data import DataLoaders; print(DataLoaders.__annotations__)"` affiche les trois noms.

#### Définition de terminé

- [ ] attributs exacts.
- [ ] dataclass immuable.
- [ ] annotations correctes.

#### Indices

1. C'est un conteneur, pas une factory.
2. Le type vient de torch.utils.data.
3. Les noms portent la sémantique du split.

---

### Exercice 2.6 — `make_dataloaders`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/data.py`. Sources : `torchvision.datasets.CIFAR10` et les classes précédentes.

#### Contexte

Cette factory est l'unique endroit où CIFAR-10, le split et les wrappers de vues sont assemblés. Les deux architectures doivent recevoir les mêmes indices.

#### Objectif

Télécharger ou ouvrir CIFAR-10, créer les trois datasets sans labels et leurs loaders.

#### Entrées

`root`, configuration data, `split_seed`, transforms train et évaluation, booléen download.

#### Sorties

`DataLoaders` : train mélangé/drop_last, validation et test ordonnés.

#### Comportement attendu

- ouvrir CIFAR train sans transform.
- construire le split 45k/5k avec Subset.
- ouvrir CIFAR test séparément.
- envelopper train en PairedViewDataset et val/test en deterministic.
- créer les loaders avec batch size/workers communs.

#### Exemple

Avec batch 256, le premier batch train contient deux tenseurs `(256,3,32,32)` ; validation/test ne contiennent aucun label.

#### Contraintes

- shuffle seulement au train.
- drop_last seulement au train.
- mêmes indices indépendamment de head_type.
- test exactement 10 000.

#### Code de départ

```python
def make_dataloaders(cfg: DictConfig, train_transform: Callable, eval_transform: Callable) -> DataLoaders:
    ...
```

#### Étapes conseillées

1. charge les deux objets CIFAR.
2. fabrique et applique les Subset.
3. enveloppe chaque split.
4. construit trois DataLoader.
5. retourne la dataclass.

#### Vérification manuelle

Après téléchargement autorisé : `PYTHONPATH=src python -c "from comparison.data import make_dataloaders; print('factory importée')"`. Le premier vrai appel doit afficher deux vues et aucun label.

#### Définition de terminé

- [ ] tailles 45k/5k/10k.
- [ ] shuffle/drop_last corrects.
- [ ] batchs constitués de deux tenseurs.
- [ ] aucune branche sur la tête.

#### Indices

1. Subset conserve l'accès au dataset source.
2. Le test officiel ne doit pas être redécoupé.
3. Le sampler aléatoire du train est indépendant du split déterministe.

