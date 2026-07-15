# EB-JEPA CIFAR-10 — Roadmap de copie et d’adaptation

> Cette roadmap est un sujet d’implémentation personnelle au format
> LeetCode/Deep-ML. Chaque exercice explique le problème, indique le fichier
> source officiel et donne le contrat attendu sans fournir le corps de solution.

## Format Deep-ML utilisé

La structure ci-dessous reprend celle de la banque officielle Deep-ML : une
description, une section d’apprentissage, un starter code, un exemple avec son
raisonnement et des cas d’acceptation. Dans ce document :

- **Problème** correspond à `description.md` ;
- **À comprendre avant de coder** correspond à `learn.md` ;
- le bloc Python est le `starter_code.py` : tu conserves la signature et
  remplaces uniquement `...` ;
- **Exemple raisonné** correspond à `example.json` ;
- **Cas d’acceptation** correspond aux comportements vérifiés normalement par
  Deep-ML, mais aucun fichier de test ne doit être créé ici ;
- **Source** indique le code que tu as le droit de consulter et copier.

Un exercice est terminé seulement lorsque tous ses cas d’acceptation sont vrais.

**But :** comparer le projecteur MLP officiel de l’exemple Image JEPA à
`FixedTreePredictor` sur CIFAR-10 avec le même ResNet-18, VICReg et le même
protocole de calcul.

## Point scientifique à retenir

Le dépôt EB-JEPA décrit explicitement son exemple Image JEPA comme une **JEA** :
il n’existe pas de prédicteur contextuel. Le composant placé après ResNet-18 est
un projecteur MLP. Dans ce projet, le mot « tête » désigne donc soit ce projecteur
MLP, soit l’arbre cortical. Tu compares deux têtes de représentation, pas deux
prédicteurs I-JEPA.

Le dépôt officiel publie une accuracy obtenue avec un linear probe. Tu as exclu
ce probe et tous les labels : ton score principal sera donc la loss VICReg sur
le split test, accompagnée d’un diagnostic d’effondrement. Ce score ne doit pas
être comparé directement aux accuracies du README officiel.

## Référence figée

- dépôt : `https://github.com/facebookresearch/eb_jepa.git` ;
- clone intact : `external/eb_jepa/` ;
- branche : `main` ;
- commit : `966e61e9285b3a876f49b9774e9720d9a99a7925` ;
- licence : Apache-2.0, dans `external/eb_jepa/LICENSE.md` ;
- registre des copies : `eb_jepa_cifar10_comparison/SOURCES.md`.

Ne modifie jamais `external/eb_jepa/`. Travaille uniquement dans
`eb_jepa_cifar10_comparison/`.

## Arborescence finale

```text
eb_jepa_cifar10_comparison/
├── LICENSE.md
├── README.md
├── SOURCES.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   └── cortical.yaml
├── src/
│   └── comparison/
│       ├── __init__.py
│       ├── augmentations.py
│       ├── data.py
│       ├── backbone.py
│       ├── heads.py
│       ├── losses.py
│       ├── optim.py
│       ├── config.py
│       ├── checkpoint.py
│       ├── train.py
│       ├── evaluate.py
│       ├── benchmark.py
│       ├── report.py
│       └── cortical/
│           ├── __init__.py
│           ├── tree_structure.py
│           ├── modulation.py
│           ├── column.py
│           ├── decomposition.py
│           ├── integration.py
│           └── fixed_tree_predictor.py
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── benchmark.py
    └── report.py
```

## Règle d’attribution

En tête de chaque fichier contenant du code EB-JEPA copié, ajoute un commentaire
avec le dépôt, le commit, le chemin source et la liste des modifications. Copie
également `external/eb_jepa/LICENSE.md` vers la racine du nouveau projet. Quand
tu modifies un fichier repris, indique clairement qu’il a été modifié, comme le
demande la licence Apache-2.0.

## Protocole commun

- CIFAR-10 : 45 000 train, 5 000 validation, 10 000 test ;
- aucun label dans le modèle, la loss ou le score ;
- ResNet-18 officiel adapté à CIFAR-10, sortie 512 ;
- tête baseline : MLP officiel `512 → 2048 → 2048 → 2048` ;
- tête corticale : `FixedTreePredictor`, entrée 512, sortie 2048 ;
- batch size 256, 300 epochs, seeds 1, 1000 et 10000 ;
- LARS, LR 0,3, warmup 10 epochs, weight decay `1e-4` ;
- VICReg : invariance 1, variance 1, covariance 80 ;
- BF16 et benchmark final sur un A100 ;
- le split test n’est lu qu’après le choix du checkpoint ;
- aucun linear probe, aucun dossier `tests/`, aucune dépendance `pytest`.

---

## Phase 0 — Préparer le projet autonome

### Exercice 0.1 — Licence et provenance

**Difficulté :** facile. **Catégorie :** ingénierie et reproductibilité.

**Problème.** Rends le nouveau projet redistribuable tout en conservant la
traçabilité du code officiel. Copie `external/eb_jepa/LICENSE.md` à la racine,
puis complète `SOURCES.md` à chaque nouvelle copie de symbole.

**Fichiers :** `LICENSE.md`, `SOURCES.md`.

**Sortie attendue :** chaque symbole repris peut être relié à un chemin et au
commit officiel sans consulter l’historique de conversation.

**À comprendre avant de coder.** Apache-2.0 autorise la copie et la modification,
mais impose de conserver la licence et d’indiquer les fichiers modifiés. Le hash
du commit rend la source reproductible même si la branche `main` évolue.

**Exemple raisonné.** Pour `VICRegLoss`, la ligne de registre doit relier le
symbole local à `eb_jepa/losses.py:306-342`, au commit figé, et indiquer
« copié avec attribution ». Ainsi, une autre personne retrouve exactement le
code de départ.

**Cas d’acceptation :** `LICENSE.md` est identique byte pour byte à la source ;
le dépôt, la branche, le commit et la licence figurent dans `SOURCES.md` ; aucune
fonction n’est déclarée comme copiée avant d’exister localement.

### Exercice 0.2 — `pyproject.toml`

**Difficulté :** facile. **Catégorie :** packaging Python.

**Problème.** Déclare un package Python 3.12 autonome nommé
`eb-jepa-cifar10-comparison`. Reprends uniquement les dépendances nécessaires à
l’exemple image : PyTorch, torchvision, OmegaConf, PyYAML, tqdm et fvcore.
N’ajoute ni `pytest`, ni les dépendances vidéo, planning ou linear probe.

**Fichier :** `eb_jepa_cifar10_comparison/pyproject.toml`.

**À comprendre avant de coder.** Le layout `src/` empêche d’importer par erreur
le code depuis le répertoire courant. PyTorch 2.6.0 et torchvision 0.21.0 forment
le couple utilisé par la référence choisie.

**Entrée / sortie.** Il n’y a pas d’entrée runtime. Le résultat est un projet
installable exposant les packages trouvés sous `src/comparison`.

**Exemple raisonné.** Après lecture TOML, `project.name` vaut
`eb-jepa-cifar10-comparison`, `requires-python` vaut `==3.12.*` et la liste des
dépendances ne contient ni `wandb`, ni `pytest`, ni dépendance vidéo.

**Cas d’acceptation :** TOML syntaxiquement valide ; exactement six dépendances
runtime demandées ; backend setuptools ; découverte des packages depuis `src`.

### Exercice 0.3 — Exports publics

**Difficulté :** facile. **Catégorie :** API Python.

**Problème.** Crée les deux `__init__.py` et limite les exports publics aux
objets utilisés par les scripts : modèle commun, fabriques de tête, VICReg et
chargement de configuration.

**Fichiers :** `src/comparison/__init__.py`, `src/comparison/cortical/__init__.py`.

**À comprendre avant de coder.** `__all__` définit l’API stable destinée aux
scripts. Comme les modules des phases suivantes n’existent pas encore, un import
paresseux permet d’importer le package sans fabriquer de fausses implémentations.

**Exemple raisonné.** `import comparison` réussit dès la phase 0 ; consulter
`comparison.__all__` retourne `ImageSSL`, `build_head`, `build_model`,
`VICRegLoss` et `load_config`. L’accès à `ImageSSL` ne devient possible qu’après
sa phase d’implémentation.

**Cas d’acceptation :** aucun symbole interne n’est public ; un nom inconnu lève
`AttributeError` ; `comparison.cortical.__all__` contient uniquement
`FixedTreePredictor`.

---

## Phase 1 — Copier les augmentations officielles

**Fichier local :** `src/comparison/augmentations.py`.

**Source :** `external/eb_jepa/examples/image_jepa/dataset.py:10-96`.

### À comprendre avant de coder

Chaque transformation reçoit une image PIL et renvoie une image PIL. Seuls
`ToTensor` et `Normalize`, placés à la fin de la composition, produisent le
tenseur `(3, 32, 32)`. Toutes les décisions aléatoires utilisent `torch.rand`,
ce qui permet à `setup_seed` de contrôler le pipeline.

### Exercice 1.1 — `RandomResizedCrop`

**Problème.** Copie la classe officielle qui applique un crop redimensionné
32×32 avec `scale=(0.2, 1.0)`. Conserve son comportement et ajoute seulement les
annotations de types manquantes.

```python
class RandomResizedCrop:
    def __init__(self, size: int, scale: tuple[float, float] = (0.2, 1.0)) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

### Exercice 1.2 — `ColorJitter`

**Problème.** Copie l’enveloppe officielle autour de
`torchvision.transforms.ColorJitter`. Le jitter est appliqué avec probabilité
0,8 et conserve les coefficients officiels 0,4/0,4/0,2/0,1.

```python
class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

### Exercice 1.3 — `Grayscale`

**Problème.** Copie la transformation aléatoire en niveaux de gris. Elle doit
maintenir trois canaux et s’activer avec la probabilité officielle 0,2.

```python
class Grayscale:
    def __init__(self, prob: float = 0.2) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

### Exercice 1.4 — `Solarization`

**Problème.** Copie la solarisation officielle avec seuil 128 et probabilité
0,1. N’ajoute pas de blur : il n’existe pas dans le pipeline de ce commit.

```python
class Solarization:
    def __init__(self, prob: float = 0.1) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

### Exercice 1.5 — `HorizontalFlip`

**Problème.** Copie le flip horizontal et conserve sa probabilité 0,5. Utilise
le RNG PyTorch comme dans la source afin que `setup_seed` contrôle les vues.

```python
class HorizontalFlip:
    def __init__(self, prob: float = 0.5) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

### Exercice 1.6 — `get_train_transforms`

**Problème.** Recopie exactement l’ordre des augmentations officielles, puis
`ToTensor` et la normalisation CIFAR-10. Les deux architectures doivent appeler
la même instance de cette fabrique.

```python
def get_train_transforms() -> transforms.Compose: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 1

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 1.1 | Une image 32×32 peut être recadrée sur 20 % à 100 % de sa surface, puis revient en 32×32. | `size=32`, scale officiel, aucune normalisation dans cette classe. |
| 1.2 | Avec `prob=0`, l’objet rendu est inchangé ; avec décision vraie, les quatre composantes du jitter sont appliquées ensemble. | RNG PyTorch, coefficients 0.4/0.4/0.2/0.1, probabilité 0.8 par défaut. |
| 1.3 | Une image RGB transformée reste RGB grâce à `num_output_channels=3`. | Probabilité 0.2 ; aucune modification quand le tirage échoue. |
| 1.4 | Un pixel 200 est solarisé autour du seuil 128 seulement lorsque le tirage réussit. | Seuil 128, probabilité 0.1, aucun blur ajouté. |
| 1.5 | Le bord gauche devient le bord droit après flip. | Probabilité 0.5 et `transforms.functional.hflip`. |
| 1.6 | Une image PIL produit un tenseur float32 `(3,32,32)` normalisé. | Ordre exact crop→jitter→gray→solarize→flip→tensor→normalize ; moyennes et écarts-types CIFAR-10 officiels. |

---

## Phase 2 — Adapter les datasets et les splits

**Fichier local :** `src/comparison/data.py`.

### À comprendre avant de coder

Le dataset CIFAR-10 retourne `(image, label)`, mais l’objectif auto-supervisé ne
doit exposer que deux vues. Le train utilise des vues réellement aléatoires ; la
validation et le test reconstruisent les mêmes vues pour rendre les checkpoints
et architectures comparables.

### Exercice 2.1 — `PairedViewDataset`

**Problème.** Copie le principe de `ImageDataset`, mais supprime le label de la
valeur retournée. Pour chaque image, produis exactement deux vues indépendantes.

**Source :** adapte `ImageDataset` depuis
`external/eb_jepa/examples/image_jepa/dataset.py:99-113`.

```python
class PairedViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable, num_crops: int = 2) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...
```

### Exercice 2.2 — `SplitIndices`

**Problème.** Représente le split déterministe 45 000/5 000 du train officiel.
Cette structure est nouvelle car l’exemple EB-JEPA utilise les 50 000 images
pour le train et le test CIFAR-10 pour son linear probe.

```python
@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]
```

### Exercice 2.3 — `build_split_indices`

**Problème.** Mélange les 50 000 indices avec un `torch.Generator` local et une
seed fixe, puis retourne deux listes disjointes de tailles 45 000 et 5 000.

```python
def build_split_indices(size: int, train_size: int, seed: int) -> SplitIndices: ...
```

### Exercice 2.4 — `DeterministicPairDataset`

**Problème.** Crée des paires de vues reproductibles pour la validation et le
test. Dérive deux seeds de `(base_seed, pair_index, image_index)` et restaure les
états RNG après chaque transformation.

```python
class DeterministicPairDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable, base_seed: int, pair_index: int) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...
```

### Exercice 2.5 — `DataLoaders`

**Problème.** Regroupe les trois DataLoaders afin que `train.py` ne reconstruise
jamais les splits. Le test loader doit être déterministe et ne doit pas shuffler.

```python
@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
```

### Exercice 2.6 — `make_dataloaders`

**Problème.** Télécharge CIFAR-10, construit le split, enveloppe chaque partie
avec le dataset approprié puis crée des loaders identiques pour les deux têtes.

```python
def make_dataloaders(cfg: DictConfig, pair_index: int = 0) -> DataLoaders: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 2

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 2.1 | Pour `(image, 7)`, deux appels indépendants à `transform(image)` donnent `(view1, view2)` et le `7` disparaît. | Longueur inchangée, exactement deux tenseurs, `num_crops != 2` refusé. |
| 2.2 | Un objet contient 45 000 indices train et 5 000 validation. | Dataclass figée, listes sans intersection. |
| 2.3 | Deux appels avec seed 17 retournent les mêmes listes ; seed 18 change l’ordre. | Couverture exacte de `range(size)`, pas de doublon, tailles invalides refusées. |
| 2.4 | `(base_seed=1,pair=0,index=42)` rend toujours la même paire ; changer `pair` change les vues. | RNG global restauré après l’appel, deux seeds différentes pour les deux vues. |
| 2.5 | `loaders.train` shuffle, `validation` et `test` non. | Les trois loaders existent et partagent batch size/workers configurés. |
| 2.6 | CIFAR train est découpé 45k/5k et CIFAR test conserve 10k images. | `drop_last=True` seulement au train ; labels absents des trois sorties ; mêmes indices pour les deux têtes. |

---

## Phase 3 — Copier le backbone et le projecteur baseline

**Fichiers locaux :** `src/comparison/backbone.py`, `src/comparison/heads.py`.

### À comprendre avant de coder

Le backbone transforme une image en feature 512. La tête transforme ensuite la
feature en représentation 2048 sur laquelle VICReg est calculée. Le baseline
doit rester identique au projecteur du commit officiel.

### Exercice 3.1 — `ResNet18`

**Problème.** Copie la classe officielle : ResNet-18 torchvision, `fc` remplacée
par `Identity`, convolution initiale 3×3 stride 1 padding 2 et aucun max-pool.
La sortie doit rester de dimension 512.

**Source :** `external/eb_jepa/examples/image_jepa/main.py:61-75`.

```python
class ResNet18(nn.Module):
    features_dim: int = 512
    def __init__(self) -> None: ...
    def forward(self, images: Tensor) -> Tensor: ...
```

### Exercice 3.2 — `MLPProjector`

**Problème.** Transforme le bloc séquentiel officiel en classe nommée sans
changer ses couches : trois Linear, deux BatchNorm1d et deux ReLU.

**Source :** extrais uniquement `ImageSSL.projector` depuis
`external/eb_jepa/examples/image_jepa/main.py:78-102`.

```python
class MLPProjector(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 2048) -> None: ...
    def forward(self, features: Tensor) -> Tensor: ...
```

### Ce que signifie « même contrat »

Tu n’as rien à coder ici. N’ajoute ni `Protocol`, ni classe abstraite, ni classe
parente commune. `MLPProjector` et `CorticalHead` héritent déjà de `nn.Module` et
PyTorch sait appeler leur méthode `forward` avec la syntaxe `head(features)`.

Dans la suite du projet, les deux têtes doivent simplement respecter la même
convention :

```text
entrée  : features, tenseur de forme (batch_size, 512)
sortie  : projections, tenseur de forme (batch_size, 2048)
```

Exemple avec un batch de huit images :

```python
features = torch.randn(8, 512)
projections = head(features)

# La forme attendue est (8, 2048), quelle que soit la tête choisie.
print(projections.shape)
```

Ce contrat sera utilisé concrètement dans l’exercice `ImageSSL`, où `head` sera
simplement annoté comme un `nn.Module`. Tu n’as donc pas besoin de comprendre ou
d’utiliser `typing.Protocol` pour terminer le projet.

### Exemples raisonnés et cas d’acceptation — Phase 3

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 3.1 | Une entrée `(8,3,32,32)` produit `(8,512)`. | `fc=Identity`, `maxpool=Identity`, conv 3×3 stride 1 padding 2, aucun poids préentraîné. |
| 3.2 | Une feature `(8,512)` traverse Linear→BN→ReLU→Linear→BN→ReLU→Linear et donne `(8,2048)`. | Trois Linear, deux BN, deux ReLU, aucune activation finale. |

---

## Phase 4 — Copier le prédicteur cortical local

Ne copie pas `Predictor/tests/`, ses caches Python, ni `JEPAPredictorLoss` : la
comparaison utilise VICReg officielle.

### À comprendre avant de coder

L’arbre est construit en largeur. Chaque colonne encode son entrée, reçoit
éventuellement contexte latéral et feedback, puis transmet une représentation à
ses enfants. Les feuilles projettent vers `dim_U`; les parents agrègent les
enfants jusqu’à la projection finale 2048.

### Exercice 4.1 — Arbre et nœuds

**Problème.** Copie `ColumnNode`, `num_columns`, `build_tree_positions`,
`level_of`, `children_of`, `parent_of` et `is_leaf` dans
`src/comparison/cortical/tree_structure.py`. Retire uniquement l’import `pdb` et
conserve l’ordre BFS ainsi que les validations de profondeur et branchement.

**Source :** `Predictor/utils/tree_structure.py`.

```python
@dataclass
class ColumnNode: ...
def num_columns(n: int, max_depth: int) -> int: ...
def build_tree_positions(n: int, max_depth: int) -> list[ColumnNode]: ...
def level_of(node: ColumnNode) -> int: ...
def children_of(node: ColumnNode) -> list[ColumnNode]: ...
def parent_of(node: ColumnNode) -> ColumnNode | None: ...
def is_leaf(node: ColumnNode, max_depth: int) -> bool: ...
```

### Exercice 4.2 — `FiLMModulation`

**Problème.** Copie la modulation FiLM dans
`src/comparison/cortical/modulation.py`. Ne change ni la LayerNorm sans paramètres
affines, ni le calcul `gamma * normalized_hidden + beta`.

**Source :** `Predictor/modules/modulation.py`.

```python
class FiLMModulation(nn.Module):
    def __init__(self, dim_h: int, dim_c: int) -> None: ...
    def forward(self, hidden: Tensor, context: Tensor) -> Tensor: ...
```

### Exercice 4.3 — Colonnes corticales

**Problème.** Copie `_lateral_attention`, `ColumnStep` et `CorticalColumn` dans
`src/comparison/cortical/column.py`. Corrige seulement l’annotation fautive
`torch.torch.Tensor` en `torch.Tensor`; ne change pas les équations.

**Source :** `Predictor/modules/column.py`.

```python
def _lateral_attention(W_q, W_k, W_v, H_self: Tensor, H_others: list[Tensor]) -> Tensor: ...
class ColumnStep(nn.Module): ...
class CorticalColumn(nn.Module): ...
```

### Exercice 4.4 — Décomposition et feedback

**Problème.** Copie `ChildDecomposition` et `FeedbackProjection` dans
`src/comparison/cortical/decomposition.py`. Chaque enfant doit conserver sa
propre couche Linear.

**Source :** `Predictor/modules/decomposition.py`.

```python
class ChildDecomposition(nn.Module): ...
class FeedbackProjection(nn.Module): ...
```

### Exercice 4.5 — Intégration récursive

**Problème.** Copie `LeafProjection`, `RecursiveIntegration` et
`LatentOutputProjection` dans `src/comparison/cortical/integration.py`. Conserve
la concaténation des enfants puis l’intégration avec l’état du parent.

**Source :** `Predictor/modules/integration.py`.

```python
class LeafProjection(nn.Module): ...
class RecursiveIntegration(nn.Module): ...
class LatentOutputProjection(nn.Module): ...
```

### Exercice 4.6 — `FixedTreePredictor`

**Problème.** Copie la classe dans
`src/comparison/cortical/fixed_tree_predictor.py`, remplace uniquement ses imports
par des imports relatifs et retire le bloc `if __name__ == "__main__"`.

**Source :** `Predictor/model/fixed_tree_predictor.py`.

```python
class FixedTreePredictor(nn.Module):
    def __init__(self, n: int, L_max: int, dim_in: int, dim_hidden: int,
                 dim_feedback: int, dim_U: int, dim_target: int,
                 disable_lateral: bool = False, disable_feedback: bool = False) -> None: ...
    def forward(self, features: Tensor) -> Tensor: ...
```

### Exercice 4.7 — `CorticalHead`

**Problème.** Écris un adaptateur minimal autour de `FixedTreePredictor`. Il ne
contient aucune couche entraînable supplémentaire et vérifie seulement les
dimensions d’entrée et de sortie.

**Fichier :** `src/comparison/heads.py`.

```python
class CorticalHead(nn.Module):
    def __init__(self, predictor: FixedTreePredictor, input_dim: int = 512, output_dim: int = 2048) -> None: ...
    def forward(self, features: Tensor) -> Tensor: ...
```

### Exercice 4.8 — `build_head`

**Problème.** Construis soit `MLPProjector`, soit `CorticalHead` à partir de la
configuration. Cette fonction est le seul branchement architectural autorisé.

```python
def build_head(cfg: DictConfig) -> nn.Module: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 4

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 4.1 | `n=2,L_max=2` crée 7 nœuds dans l’ordre root, niveau 1, niveau 2. | IDs uniques, parents/enfants cohérents, `n<2` et profondeur négative refusés. |
| 4.2 | `hidden (B,H)` et `context (B,C)` produisent `(B,H)`. | LayerNorm non affine, gamma et beta issus du contexte, aucune mutation des entrées. |
| 4.3 | Sans sœurs, l’attention latérale vaut zéro ; avec sœurs, softmax porte sur les sœurs. | Poids propres à la colonne receveuse, résiduel conservé, annotation corrigée uniquement. |
| 4.4 | Pour `n=2`, chaque module renvoie une liste de deux tenseurs issus de deux Linear différentes. | Aucun partage accidentel entre enfants, formes `(B,dim_child)` ou `(B,dim_feedback)`. |
| 4.5 | Deux `U_children (B,U)` sont concaténés en `(B,2U)`, agrégés puis combinés avec `B_v`. | Feuille `(B,H)→(B,U)`, parent `(B,H)+enfants→(B,U)`, sortie `(B,U)→(B,2048)`. |
| 4.6 | Une feature `(4,512)` parcourt récursivement tous les nœuds et produit `(4,2048)`. | Imports relatifs, un encode par colonne, flags lateral/feedback respectés, bloc de démonstration retiré. |
| 4.7 | L’adaptateur délègue directement au predictor. | Aucun Linear/BN ajouté, vérification 512 en entrée et 2048 en sortie. |
| 4.8 | `head_type=baseline` construit `MLPProjector`; `cortical` construit l’arbre configuré. | Type inconnu refusé, seul branchement architectural du projet. |

---

## Phase 5 — Modèle commun

**Fichier :** `src/comparison/heads.py`.

### À comprendre avant de coder

Le modèle commun garantit que le changement de tête ne modifie ni le backbone
ni la boucle d’entraînement. Le tuple retourné sépare les features utiles au
diagnostic des projections utilisées par VICReg.

### Exercice 5.1 — `ImageSSL`

**Problème.** Généralise la classe officielle pour recevoir une tête injectée au
lieu de construire toujours son MLP. Le forward doit garder le retour officiel
`(features, projections)`.

**Source :** adapte `external/eb_jepa/examples/image_jepa/main.py:78-102`.

```python
class ImageSSL(nn.Module):
    def __init__(self, backbone: ResNet18, head: nn.Module) -> None: ...
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]: ...
```

### Exercice 5.2 — `build_model`

**Problème.** Crée un nouveau ResNet-18 et la tête demandée, puis compose les
deux dans `ImageSSL`. Aucune autre différence ne doit dépendre de `head_type`.

```python
def build_model(cfg: DictConfig) -> ImageSSL: ...
```

### Exercice 5.3 — `ParameterCounts`

**Problème.** Compte séparément les paramètres entraînables du backbone, de la
tête et du système complet afin de mesurer l’économie structurelle.

```python
@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    head: int
    total: int

def count_parameters(model: ImageSSL) -> ParameterCounts: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 5

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 5.1 | `(B,3,32,32)` donne features `(B,512)` et projections `(B,2048)`. | Backbone appelé une fois, tête appelée sur les features, tuple officiel conservé. |
| 5.2 | Deux configs identiques sauf `head_type` créent deux ResNet-18 structurellement identiques. | Pas de condition sur la tête ailleurs, nouvelles instances sans paramètres partagés. |
| 5.3 | `backbone + head == total`. | Compter uniquement `requires_grad=True`, valeurs entières, séparation exacte des modules. |

---

## Phase 6 — Copier VICReg officielle

**Fichier local :** `src/comparison/losses.py`.

**Source :** `external/eb_jepa/eb_jepa/losses.py`.

### À comprendre avant de coder

VICReg combine trois forces : rapprocher les deux vues, maintenir une variance
minimale par dimension et décorréler les dimensions. Les composantes doivent
rester des tenseurs différentiables jusqu’au backward.

### Exercice 6.1 — `HingeStdLoss`

**Problème.** Copie la classe des lignes 56-81, y compris le centrage,
`var(dim=0)`, epsilon `0.0001`, marge 1 et hinge moyen.

```python
class HingeStdLoss(nn.Module):
    def __init__(self, std_margin: float = 1.0) -> None: ...
    def forward(self, representations: Tensor) -> Tensor: ...
```

### Exercice 6.2 — `CovarianceLoss`

**Problème.** Copie la classe des lignes 84-111, y compris sa méthode
`off_diagonal` et la normalisation officielle par la moyenne des termes hors
diagonale.

```python
class CovarianceLoss(nn.Module):
    def off_diagonal(self, matrix: Tensor) -> Tensor: ...
    def forward(self, representations: Tensor) -> Tensor: ...
```

### Exercice 6.3 — `VICRegLoss`

**Problème.** Copie la classe des lignes 306-342 sans changer la pondération :
MSE coefficient 1, somme des deux variance losses coefficient 1 et somme des
deux covariance losses coefficient 80 via la configuration.

```python
class VICRegLoss(nn.Module):
    def __init__(self, std_coeff: float = 1.0, cov_coeff: float = 80.0) -> None: ...
    def forward(self, z1: Tensor, z2: Tensor) -> dict[str, Tensor]: ...
```

### Exercice 6.4 — `CollapseDiagnostics`

**Problème.** Ajoute un diagnostic absent du dépôt officiel. Mesure l’écart-type
moyen et la fraction des dimensions dont l’écart-type dépasse un seuil, puis
signale un effondrement si cette fraction est trop faible.

```python
@dataclass(frozen=True)
class CollapseDiagnostics:
    mean_std: float
    active_dimension_fraction: float
    collapsed: bool

def diagnose_collapse(representations: Tensor, std_threshold: float = 1e-2,
                      active_fraction_threshold: float = 0.1) -> CollapseDiagnostics: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 6

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 6.1 | Des colonnes constantes ont std 0 et coût 1 ; une std ≥1 a coût 0. | Centrage, variance sur batch, epsilon `1e-4`, moyenne des hinges. |
| 6.2 | Pour l’identité, tous les termes hors diagonale sont zéro. | Matrice carrée exigée, covariance divisée par `B-1`, moyenne des carrés hors diagonale. |
| 6.3 | Si `z1==z2`, invariance=0 mais variance/covariance restent calculées. | Clés exactes `loss`, `invariance_loss`, `var_loss`, `cov_loss`; total `sim + 1*var + 80*cov`. |
| 6.4 | Un tenseur tout-zéro donne fraction active 0 et `collapsed=True`. | Calcul sans gradient, statistiques Python sérialisables, seuils appliqués explicitement. |

---

## Phase 7 — Copier l’optimisation officielle

**Fichier local :** `src/comparison/optim.py`.

### À comprendre avant de coder

LARS adapte la taille du pas à la norme de chaque paramètre. Les biais et
normalisations sont exclus de cette adaptation. Le scheduler modifie le LR par
epoch : warmup linéaire, puis décroissance cosinus.

### Exercice 7.1 — `LARS`

**Problème.** Copie intégralement l’optimiseur officiel, y compris momentum,
adaptation locale, clipping et exclusion des biais/normalisations. N’en remplace
pas le comportement par un optimiseur de bibliothèque différent.

**Source :** `external/eb_jepa/examples/image_jepa/main.py:105-207`.

```python
class LARS(torch.optim.Optimizer): ...
```

### Exercice 7.2 — `WarmupCosineScheduler`

**Problème.** Copie le scheduler officiel et ajoute seulement `state_dict` et
`load_state_dict` pour les checkpoints. Le warmup commence à `3e-5`, atteint
0,3 à la dixième epoch puis suit le cosinus jusqu’à zéro.

**Source :** `external/eb_jepa/examples/image_jepa/main.py:210-247`.

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr,
                 min_lr=0.0, warmup_start_lr=3e-5) -> None: ...
    def step(self, epoch: int) -> None: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 7

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 7.1 | Un paramètre sans gradient ne change pas ; un vecteur 1D exclu n’est ni adapté ni weight-decay. | Code officiel complet, momentum 0.9, eta 0.02 lors de la construction finale, clipping actif. |
| 7.2 | Epoch 0→`3e-5`, epoch 9→`0.3`, epoch 299 proche de 0. | Tous les groupes reçoivent le même LR, état sérialisable, reprise sans décalage d’epoch. |

---

## Phase 8 — Configuration

**Fichier local :** `src/comparison/config.py`.

### À comprendre avant de coder

OmegaConf fusionne le YAML et les overrides CLI. La validation doit échouer tôt :
une expérience invalide ne doit pas consommer plusieurs heures de GPU avant de
révéler une dimension ou un coefficient différent.

### Exercice 8.1 — `load_config`

**Problème.** Copie le chargement OmegaConf avec overrides en notation pointée,
puis appelle une validation locale. Le fichier YAML reste la source unique des
hyperparamètres.

**Source :** `external/eb_jepa/eb_jepa/training_utils.py:241-269`.

```python
def load_config(path: str | Path, overrides: dict[str, object] | None = None) -> DictConfig: ...
```

### Exercice 8.2 — `validate_config`

**Problème.** Vérifie les invariants du protocole : dimensions 512/2048, batch
256, VICReg 1/80, 300 epochs, seeds autorisées, split 45k/5k/10k et paramètres
corticaux positifs. Refuse toute clé `linear_probe` activée.

```python
def validate_config(cfg: DictConfig) -> None: ...
```

### Exercice 8.3 — YAML baseline et cortical

**Problème.** Copie les valeurs officielles utiles dans `configs/baseline.yaml`,
supprime BCS, sweep, W&B et linear probe, puis duplique le fichier en
`configs/cortical.yaml`. Seuls le nom, le dossier de sortie, `head_type` et le
bloc de dimensions corticales peuvent différer.

**Source :** pars de
`external/eb_jepa/examples/image_jepa/cfgs/default.yaml`.

### Exemples raisonnés et cas d’acceptation — Phase 8

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 8.1 | L’override `optim.epochs=2` remplace seulement cette valeur avant validation. | Fichier absent→`FileNotFoundError`, overrides pointés, retour `DictConfig` résolu. |
| 8.2 | `output_dim=1024` ou `cov_coeff=79` est refusé avec le chemin de clé dans le message. | Tous les invariants globaux contrôlés, valeurs positives, aucun linear probe actif. |
| 8.3 | Un diff des YAML ne montre que identité du run, sortie, tête et paramètres corticaux. | Baseline fidèle au default officiel ; blocs data/loss/optim/training communs textuellement identiques. |

---

## Phase 9 — Checkpoints et utilitaires officiels

**Fichier local :** `src/comparison/checkpoint.py`.

### À comprendre avant de coder

Une reprise fidèle nécessite les poids mais aussi optimiseur, scheduler, scaler,
epoch et configuration. La compatibilité doit être contrôlée avant
`load_state_dict`, sinon le modèle peut être partiellement muté avant l’erreur.

### Exercice 9.1 — `setup_device` et `setup_seed`

**Problème.** Copie les deux fonctions. Ajoute uniquement la seed des workers via
le générateur du DataLoader; ne change pas les règles CPU/CUDA officielles.

**Source :** `external/eb_jepa/eb_jepa/training_utils.py:18-35`.

```python
def setup_device(device: str = "auto") -> torch.device: ...
def setup_seed(seed: int) -> None: ...
```

### Exercice 9.2 — `save_checkpoint`

**Problème.** Copie la sauvegarde officielle et rends obligatoires le scheduler,
le scaler, la configuration résolue, `head_type` et la meilleure validation.

**Source :** adapte `external/eb_jepa/eb_jepa/training_utils.py:146-176`.

```python
def save_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer,
                    scheduler: WarmupCosineScheduler, scaler: GradScaler,
                    epoch: int, cfg: DictConfig, best_validation: float) -> None: ...
```

### Exercice 9.3 — `load_checkpoint`

**Problème.** Recharge tous les états officiels et vérifie que `head_type` et les
dimensions du checkpoint correspondent à la configuration actuelle avant de
modifier le modèle.

**Source :** adapte `external/eb_jepa/eb_jepa/training_utils.py:179-238`.

```python
def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer,
                    scheduler: WarmupCosineScheduler, scaler: GradScaler,
                    cfg: DictConfig, device: torch.device) -> dict[str, object]: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 9

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 9.1 | Seed 1 reproduit initialisation et ordre du split ; `auto` choisit CUDA si disponible. | Python, NumPy, torch CPU et tous GPU initialisés ; device explicite respecté. |
| 9.2 | Après epoch 49, le fichier contient `epoch=49`, états complets et config résolue. | Dossier créé, clés obligatoires présentes, aucune donnée de linear probe. |
| 9.3 | Charger un checkpoint baseline dans cortical échoue avant mutation. | `map_location` correct, préfixe `_orig_mod.` géré, retour reprend à `epoch+1`, états restaurés. |

---

## Phase 10 — Adapter l’entraînement officiel

**Fichier local :** `src/comparison/train.py`.

### À comprendre avant de coder

Chaque batch contient deux vues. Le même modèle produit `z1` et `z2`, VICReg
calcule la seule loss rétropropagée, puis LARS met les paramètres à jour. La
validation choisit le checkpoint sans consulter le test.

### Exercice 10.1 — `train_epoch`

**Problème.** Pars du corps officiel puis retire tous les arguments, calculs et
métriques liés à `LinearProbe`, `target`, cross-entropy et accuracy. La seule loss
rétropropagée est `loss_dict["loss"]`. Conserve autocast, GradScaler, LARS, tqdm
et l’agrégation dynamique des composantes VICReg.

**Source :** adapte
`external/eb_jepa/examples/image_jepa/main.py:250-332`.

```python
def train_epoch(model: ImageSSL, train_loader: DataLoader, optimizer: LARS,
                scheduler: WarmupCosineScheduler, scaler: GradScaler,
                device: torch.device, epoch: int, loss_fn: VICRegLoss,
                use_amp: bool = True, dtype: torch.dtype = torch.bfloat16,
                max_batches: int | None = None) -> dict[str, float]: ...
```

### Exercice 10.2 — `evaluate_validation`

**Problème.** Ajoute une boucle sans gradient utilisant exactement VICReg et les
vues déterministes. Elle retourne la loss moyenne pondérée par le nombre
d’images et ne sélectionne jamais un checkpoint à partir du test.

```python
@torch.no_grad()
def evaluate_validation(model: ImageSSL, loader: DataLoader, loss_fn: VICRegLoss,
                        device: torch.device, use_amp: bool,
                        dtype: torch.dtype) -> dict[str, float]: ...
```

### Exercice 10.3 — `RunResult`

**Problème.** Représente le résultat observable d’un run sans logs W&B : chemin
du meilleur checkpoint, dernière epoch, meilleure loss validation et durée.

```python
@dataclass(frozen=True)
class RunResult:
    best_checkpoint: Path
    last_epoch: int
    best_validation_loss: float
    duration_seconds: float
```

### Exercice 10.4 — `run`

**Problème.** Réutilise l’ordre officiel configuration→device→seed→data→modèle→
optimisation→epochs→checkpoint, mais utilise tes fabriques communes et ta
validation sans labels. Les deux têtes traversent exactement cette fonction.

**Source :** adapte l’orchestration de
`external/eb_jepa/examples/image_jepa/main.py:335-fin`.

```python
def run(config_path: str | Path, overrides: dict[str, object] | None = None,
        resume: str | Path | None = None, max_batches: int | None = None) -> RunResult: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 10

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 10.1 | Deux batches produisent quatre composantes finies et modifient au moins un paramètre. | Aucun `target`, probe ou cross-entropy ; autocast BF16 ; limite `max_batches` exacte ; scheduler une fois par epoch. |
| 10.2 | Un dernier batch plus petit pèse proportionnellement moins dans la moyenne. | `model.eval`, `no_grad`, aucune mutation, dictionnaire des quatre losses. |
| 10.3 | Un run interrompu garde le chemin du meilleur checkpoint et sa loss. | Dataclass figée, chemins `Path`, durée non négative, epoch entière. |
| 10.4 | Baseline et cortical exécutent la même séquence ; seul `build_head` diffère. | Reprise supportée, meilleur checkpoint basé sur validation, dernier checkpoint périodique, test jamais chargé. |

---

## Phase 11 — Score test sans labels

**Fichier local :** `src/comparison/evaluate.py`.

### À comprendre avant de coder

Le score test doit être robuste au hasard des augmentations : cinq paires fixes
sont évaluées pour chaque seed et chaque tête. L’effondrement est une condition
d’échec indépendante de la valeur moyenne de la loss.

### Exercice 11.1 — `PairEvaluation`

**Problème.** Stocke la loss VICReg et ses composantes pour une paire
déterministe complète du split test, avec son diagnostic d’effondrement.

```python
@dataclass(frozen=True)
class PairEvaluation:
    pair_index: int
    loss: float
    invariance_loss: float
    var_loss: float
    cov_loss: float
    collapse: CollapseDiagnostics
```

### Exercice 11.2 — `evaluate_pair`

**Problème.** Parcours un `DeterministicPairDataset`, calcule les métriques
pondérées par image et concatène les projections CPU nécessaires au diagnostic.

```python
@torch.no_grad()
def evaluate_pair(model: ImageSSL, loader: DataLoader, loss_fn: VICRegLoss,
                  device: torch.device, pair_index: int) -> PairEvaluation: ...
```

### Exercice 11.3 — `TestEvaluation`

**Problème.** Agrège cinq évaluations déterministes et expose moyenne, écart-type
et nombre de paires effondrées. Ce résultat constitue le score principal.

```python
@dataclass(frozen=True)
class TestEvaluation:
    head_type: str
    seed: int
    pairs: list[PairEvaluation]
    loss_mean: float
    loss_std: float
    collapsed_pairs: int
```

### Exercice 11.4 — `evaluate_test`

**Problème.** Recharge le meilleur checkpoint, construit successivement les cinq
test loaders avec `pair_index=0..4`, appelle `evaluate_pair` et agrège sans
modifier le modèle.

```python
def evaluate_test(cfg: DictConfig, checkpoint_path: str | Path,
                  num_pairs: int = 5) -> TestEvaluation: ...
```

### Exercice 11.5 — `write_evaluation_json`

**Problème.** Sérialise le résultat, la configuration et le hash du checkpoint
dans un JSON ne contenant que des types natifs.

```python
def write_evaluation_json(path: str | Path, cfg: DictConfig,
                          result: TestEvaluation, checkpoint_sha256: str) -> None: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 11

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 11.1 | La paire 2 stocke ses quatre losses et le diagnostic calculé sur ses projections. | Types natifs sauf dataclass imbriquée, index conservé, aucune feature brute sauvegardée. |
| 11.2 | Deux batches 256 et 16 sont agrégés par 272 images, pas par deux moyennes égales. | `no_grad`, projections déplacées CPU, modèle inchangé, diagnostic sur l’ensemble complet. |
| 11.3 | Losses `[1,2,3,4,5]` donnent moyenne 3 et écart-type défini par le protocole. | Exactement cinq paires, compteur d’effondrement exact, head_type/seed conservés. |
| 11.4 | Les pair_index 0..4 réutilisent le même checkpoint et le même dataset test. | Meilleur checkpoint validation uniquement, ordre déterministe, aucune mise à jour de poids. |
| 11.5 | Relire le JSON avec `json.loads` ne rencontre ni Tensor, ni Path, ni DictConfig. | Config résolue, SHA-256 présent, écriture UTF-8 indentée et dossier créé. |

---

## Phase 12 — Benchmark A100

**Fichier local :** `src/comparison/benchmark.py`.

### À comprendre avant de coder

Les opérations CUDA sont asynchrones : un chronomètre CPU sans synchronisation
mesure surtout le lancement des kernels. Les FLOPs, paramètres, latence et
mémoire répondent à des questions différentes et doivent rester séparés.

### Exercice 12.1 — `ComputeMetrics`

**Problème.** Définis les paramètres, FLOPs, latence, débit et mémoire de la tête
seule puis du modèle complet. Toutes les valeurs doivent utiliser des unités
explicites.

```python
@dataclass(frozen=True)
class ComputeMetrics:
    parameters: int
    flops_per_image: int
    median_latency_ms: float
    p95_latency_ms: float
    images_per_second: float
    peak_memory_mib: float
```

### Exercice 12.2 — `measure_flops`

**Problème.** Utilise `fvcore.nn.FlopCountAnalysis` sur des tenseurs déjà placés
sur le GPU. Mesure séparément une tête avec `(256, 512)` et le système avec
`(256, 3, 32, 32)`.

```python
def measure_flops(module: nn.Module, example_input: Tensor) -> int: ...
```

### Exercice 12.3 — `measure_latency`

**Problème.** Utilise des `torch.cuda.Event`, 100 warmups, 500 mesures et des
synchronisations explicites. Les copies CPU→GPU restent hors de la région.

```python
def measure_latency(function: Callable[[], object], warmup: int = 100,
                    iterations: int = 500) -> tuple[float, float]: ...
```

### Exercice 12.4 — `measure_peak_memory`

**Problème.** Réinitialise les statistiques CUDA juste avant le callable et
retourne le pic alloué en MiB après synchronisation.

```python
def measure_peak_memory(function: Callable[[], object]) -> float: ...
```

### Exercice 12.5 — `benchmark_model`

**Problème.** Vérifie que le GPU contient `A100`, force BF16 et produit les
métriques de la tête et du système dans les mêmes conditions.

```python
def benchmark_model(model: ImageSSL, cfg: DictConfig,
                    device: torch.device) -> dict[str, ComputeMetrics]: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 12

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 12.1 | Une tête peut avoir moins de paramètres mais une latence supérieure ; les deux valeurs restent publiées. | Unités dans les noms, valeurs finies positives, tête et système séparés. |
| 12.2 | FLOPs du système mesurés avec batch 256 sont divisés par 256 pour obtenir par image. | Entrée sur même device, modules en eval, opérateurs non supportés signalés. |
| 12.3 | 100 appels chauffent les kernels, les 500 suivants construisent médiane et p95. | Events CUDA, synchro avant/après, durées en ms, séquence vide impossible. |
| 12.4 | Les stats sont remises à zéro juste avant la région. | Synchronisation, conversion octets→MiB, aucun pic précédent réutilisé. |
| 12.5 | Les deux configs utilisent exactement `(256,512)` et `(256,3,32,32)` en BF16. | GPU dont le nom contient A100, mêmes warmups/itérations, résultat sérialisable. |

---

## Phase 13 — Rapport final

**Fichier local :** `src/comparison/report.py`.

### À comprendre avant de coder

La non-infériorité porte sur la loss test moyenne, mais la réussite exige aussi
absence d’effondrement et gain de compute. Le rapport doit montrer les données
brutes avant d’afficher le verdict.

### Exercice 13.1 — `SeedResult`

**Problème.** Associe une architecture, une seed, son `TestEvaluation` et ses
métriques de compute. Refuse les résultats dont la seed ou le type de tête ne
correspondent pas au JSON chargé.

```python
@dataclass(frozen=True)
class SeedResult:
    head_type: str
    seed: int
    evaluation: TestEvaluation
    compute: dict[str, ComputeMetrics]
```

### Exercice 13.2 — `AggregateResult`

**Problème.** Agrège exactement les seeds 1, 1000 et 10000 pour une tête et
calcule la moyenne et l’écart-type de la loss test ainsi que le nombre total de
paires effondrées.

```python
@dataclass(frozen=True)
class AggregateResult: ...

def aggregate_seed_results(results: Sequence[SeedResult]) -> AggregateResult: ...
```

### Exercice 13.3 — `ComparisonDecision`

**Problème.** Déclare la réussite seulement si la tête corticale est
non-inférieure sur la loss test, n’a aucune paire effondrée et réduit réellement
le compute de la tête. Conserve les raisons détaillées en cas d’échec.

```python
@dataclass(frozen=True)
class ComparisonDecision: ...

def compare_systems(baseline: AggregateResult, cortical: AggregateResult,
                    non_inferiority_margin: float) -> ComparisonDecision: ...
```

### Exercice 13.4 — `write_report`

**Problème.** Écris un Markdown et un JSON montrant les six runs, leurs
dispersions, les effondrements et les ratios paramètres/FLOPs/latence/mémoire.

```python
def write_report(output_dir: str | Path, baseline: AggregateResult,
                 cortical: AggregateResult,
                 decision: ComparisonDecision) -> None: ...
```

### Exemples raisonnés et cas d’acceptation — Phase 13

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 13.1 | `(baseline, seed=1)` ne peut charger un JSON cortical ou seed 1000. | Cohérence identité/JSON, compute présent, aucun doublon silencieux. |
| 13.2 | Trois seeds donnent moyenne, écart-type et somme des collapsed_pairs. | Ensemble exact `{1,1000,10000}`, même tête, résultats individuels conservés. |
| 13.3 | Cortical dans la marge mais avec un effondrement donne `success=False`. | Critères booléens séparés, ratios compute calculés, raisons explicites pour chaque échec. |
| 13.4 | Markdown contient tableau par seed, agrégats, ratios et verdict ; JSON contient les mêmes nombres. | Aucun arrondi avant calcul, données manquantes refusées, deux formats cohérents. |

---

## Phase 14 — Scripts minces

### À comprendre avant de coder

Un script CLI est un adaptateur : il transforme les chaînes de la ligne de
commande en arguments Python, appelle une fonction métier et traduit les erreurs
en code de sortie. Il ne contient aucun calcul ML.

### Exercice 14.1 — Entrées CLI

**Problème.** Dans chacun des quatre fichiers de `scripts/`, analyse uniquement
les arguments, appelle la fonction de `src/comparison/` correspondante et
retourne un code de sortie. Ne duplique aucune logique métier dans les scripts.

```python
def main(argv: Sequence[str] | None = None) -> int: ...
```

Arguments attendus :

- `scripts/train.py` : `--config`, `--resume`, `--max-batches` ;
- `scripts/evaluate.py` : `--config`, `--checkpoint`, `--output` ;
- `scripts/benchmark.py` : `--configs`, `--output` ;
- `scripts/report.py` : `--evaluations`, `--benchmark`, `--output-dir`.

### Exemple raisonné et cas d’acceptation — Phase 14

| Exercice | Exemple raisonné | Cas d’acceptation |
|---|---|---|
| 14.1 | `python scripts/train.py --config configs/baseline.yaml --max-batches 2` charge la config puis appelle `train.run(...)`. | Chaque script définit `main(argv=None)`, protège son entrée avec `if __name__ == "__main__"`, ne duplique aucune logique métier, écrit les erreurs sur stderr et retourne 0 uniquement si l’artefact demandé existe. |

---

## Contrôles courts — pas de suite de tests

Tu ne crées aucun fichier `test_*.py`. Avant l’A100, exécute seulement deux
batches de chaque configuration avec `--max-batches 2`. Les deux commandes
doivent produire les mêmes clés de métriques, des valeurs finies, une sortie de
forme `(batch, 2048)` et un checkpoint rechargeable.

## Runs finaux

Pour chaque tête, crée trois fichiers de configuration dont seules `seed` et le
dossier de sortie changent : 1, 1000 et 10000. Entraîne les six runs pendant 300
epochs, sélectionne chaque checkpoint uniquement sur la validation, puis évalue
une fois le test avec cinq paires déterministes.

Le benchmark A100 est effectué séparément de l’entraînement avec batch 256,
BF16, 100 warmups et 500 mesures. Le rapport final doit publier un verdict même
si l’hypothèse corticale échoue.

## Définition de terminé

- `external/eb_jepa/` correspond toujours au commit officiel figé ;
- chaque symbole copié apparaît dans `SOURCES.md` avec son chemin source ;
- les fichiers modifiés portent une notice d’adaptation ;
- aucune fonction EB-JEPA inutile n’est copiée ;
- aucune utilisation des labels ou du linear probe ne subsiste ;
- seul `build_head` distingue les deux architectures ;
- les six runs et le benchmark A100 sont agrégés ;
- aucun test automatisé n’a été créé.
