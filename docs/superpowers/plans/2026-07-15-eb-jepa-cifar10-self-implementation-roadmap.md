# EB-JEPA CIFAR-10 — Roadmap d'implémentation personnelle

Cette roadmap est l'index du projet. Les énoncés complets vivent dans
`docs/challenges/eb-jepa-cifar10/` : chaque exercice y est présenté comme un
problème LeetCode autonome appliqué à un véritable dépôt Python.

## But scientifique

Comparer deux têtes de représentation sur CIFAR-10 en gardant tout le reste
identique :

```text
baseline : image -> ResNet-18 -> MLP officiel -> représentation 2048
cortical : image -> ResNet-18 -> FixedTreePredictor -> représentation 2048
```

L'exemple Image d'EB-JEPA est une JEA utilisant VICReg ; il ne contient pas de
prédicteur contextuel I-JEPA. Le score principal est la loss VICReg sur le split
test, accompagnée d'un diagnostic d'effondrement. Aucun label et aucun linear
probe ne participent au protocole.

## Référence figée

- dépôt : `https://github.com/facebookresearch/eb_jepa.git` ;
- clone intact : `external/eb_jepa/` ;
- commit : `966e61e9285b3a876f49b9774e9720d9a99a7925` ;
- licence : Apache-2.0 ;
- projet à modifier : `eb_jepa_cifar10_comparison/` ;
- registre de provenance : `eb_jepa_cifar10_comparison/SOURCES.md`.

Ne modifie jamais `external/eb_jepa/`.

## Protocole commun imposé

- CIFAR-10 : 45 000 images train, 5 000 validation, 10 000 test ;
- ResNet-18 adapté aux images 32×32, sortie 512 ;
- têtes 512 vers 2048 ; batch size 256 ; 300 epochs ;
- seeds 1, 1000 et 10000 ; LARS ; LR 0,3 ; warmup 10 epochs ;
- weight decay `1e-4` ; VICReg `(1, 1, 80)` ; précision BF16 ;
- test évalué seulement après le gel de la configuration ;
- benchmark final sur A100 ;
- aucun test automatisé, dossier `tests/` ou usage de `pytest`.

## Mode d’emploi des exercices

## Racine des chemins

Tous les chemins `src/...`, `configs/...` et `scripts/...` partent de :

```text
eb_jepa_cifar10_comparison/
```

Par exemple, `src/comparison/model.py` désigne réellement
`eb_jepa_cifar10_comparison/src/comparison/model.py` depuis la racine du dépôt.

## Comment résoudre un exercice

1. Lis `Où travailler` et ouvre uniquement les fichiers indiqués.
2. Lis `Contexte` avant le code : il explique ce qui existe déjà.
3. Recopie le `Code de départ` dans le fichier indiqué si le symbole n'existe pas.
4. Remplace les `...` en suivant `Comportement attendu`.
5. Lance la `Vérification manuelle` depuis `eb_jepa_cifar10_comparison/`.
6. Compare le résultat avec la checklist `Définition de terminé`.
7. Consulte les indices dans l'ordre seulement si tu bloques.

Les commandes supposent que le projet est installé avec `pip install -e .` ou
que `PYTHONPATH=src` est défini.

## Deux types d'exercices

### Copie guidée

La solution existe dans `external/eb_jepa/` ou `Predictor/`. Tu dois retrouver
et copier les symboles indiqués. L'énoncé liste exhaustivement les modifications
autorisées ; ne refactorise pas les équations pendant la copie.

### Conception

Le composant n'existe pas tel quel dans la référence. Tu reçois son contexte,
son contrat, un exemple et un squelette, mais pas son corps complet.

## Absence de tests automatisés

Ce projet ne demande ni `pytest`, ni dossier `tests/`. Les blocs de vérification
sont des contrôles ponctuels destinés à te donner un retour immédiat. Ils ne font
pas partie du code livré.

## Conventions de tenseurs

- `B` : taille du batch ;
- images : `(B, 3, 32, 32)` ;
- features du backbone : `(B, 512)` ;
- représentations des têtes : `(B, 2048)` ;
- une vue est une augmentation aléatoire d'une image source.

## Aide

Chaque exercice contient trois indices progressifs. L'indice 1 rappelle le
concept, l'indice 2 indique la structure à employer, et l'indice 3 décrit presque
l'algorithme sans fournir le code final.


## Arborescence cible

```text
eb_jepa_cifar10_comparison/
├── LICENSE.md
├── README.md
├── SOURCES.md
├── pyproject.toml
├── configs/{baseline,cortical}.yaml
├── src/comparison/
│   ├── __init__.py
│   ├── augmentation.py
│   ├── data.py
│   ├── backbone.py
│   ├── heads.py
│   ├── model.py
│   ├── losses.py
│   ├── optim.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── report.py
│   └── cortical/
└── scripts/{train,evaluate,benchmark,report}.py
```

## Définition globale de réussite

Les six runs utilisent le même backbone, les mêmes données et les mêmes
hyperparamètres. La tête corticale réussit si son score test VICReg reste dans la
tolérance fixée face au baseline tout en réduisant le compute mesuré sur A100.
Le rapport doit aussi montrer l'absence d'effondrement et les résultats par seed.

---

# Exercices détaillés

# Phase 0 — Préparer le projet autonome

Cette phase crée le paquet Python indépendant dans lequel toute la comparaison sera écrite.

### Exercice 0.1 — `Licence et provenance`

**Difficulté :** Facile

#### Où travailler

`LICENSE.md` et `SOURCES.md`. Source : `external/eb_jepa/LICENSE.md` et les fichiers listés dans le registre.

#### Contexte

Le projet réutilise du code Apache-2.0. Une personne doit pouvoir retrouver le dépôt, le commit et le symbole source sans connaître cette conversation.

#### Objectif

Copier la licence officielle sans la modifier et créer une ligne de provenance uniquement lorsqu'un symbole existe réellement dans le projet.

#### Entrées

Aucune entrée runtime.

#### Sorties

Deux documents Markdown ; `LICENSE.md` identique à la référence et un registre lisible.

#### Comportement attendu

- copier `external/eb_jepa/LICENSE.md` vers la racine du projet.
- inscrire le commit `966e61e9285b3a876f49b9774e9720d9a99a7925` dans `SOURCES.md`.
- pour chaque copie, donner symbole local, chemin source et adaptations.

#### Exemple

`VICRegLoss | src/comparison/losses.py | eb_jepa/losses.py | copié, imports adaptés` permet de retrouver l'origine de la classe.

#### Contraintes

- ne modifie pas le texte de la licence.
- n'annonce pas un symbole qui n'existe pas encore.
- indique chaque fichier local modifié.

#### Code de départ

```python
# SOURCES.md

| Symbole local | Fichier local | Source | Adaptations |
|---|---|---|---|
| ... | ... | ... | ... |
```

#### Étapes conseillées

1. commence par la licence.
2. ajoute les métadonnées du dépôt.
3. ajoute une ligne au registre après chaque copie.

#### Vérification manuelle

`cmp LICENSE.md ../external/eb_jepa/LICENSE.md` doit terminer sans sortie.

#### Définition de terminé

- [ ] la licence est identique octet par octet.
- [ ] le commit figé est présent.
- [ ] chaque ligne décrit une copie réelle.

#### Indices

1. Une licence se copie comme un artefact, pas comme du code à reformuler.
2. Le registre est une table Markdown.
3. Une ligne correspond à un symbole ou groupe de symboles provenant du même emplacement.

---

### Exercice 0.2 — `Déclarer le package dans pyproject.toml`

**Difficulté :** Facile

#### Où travailler

`pyproject.toml` à la racine du projet de comparaison.

#### Contexte

Le code vit sous `src/comparison`. Sans configuration de packaging, Python peut importer accidentellement un autre dossier `comparison` ou ne pas trouver le package.

#### Objectif

Déclarer un projet Python 3.12 installable avec setuptools et uniquement les six dépendances nécessaires.

#### Entrées

Aucune entrée runtime ; le fichier est lu par un outil de packaging.

#### Sorties

Un document TOML valide nommé `eb-jepa-cifar10-comparison`.

#### Comportement attendu

- définir `[build-system]` avec setuptools.
- définir `[project]`, Python `==3.12.*` et les dépendances.
- configurer la découverte des packages sous `src`.

#### Exemple

`pip install -e .` doit rendre `import comparison` possible depuis un autre dossier.

#### Contraintes

- dépendances : torch, torchvision, omegaconf, pyyaml, tqdm, fvcore.
- pas de pytest, wandb ou dépendances vidéo.
- conserver le layout `src/`.

#### Code de départ

```python
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eb-jepa-cifar10-comparison"
requires-python = "==3.12.*"
dependencies = [
    # À compléter avec exactement six dépendances.
]

[tool.setuptools.packages.find]
where = ["src"]
```

#### Étapes conseillées

1. écris d'abord les métadonnées.
2. ajoute les six bibliothèques runtime.
3. configure la découverte setuptools.

#### Vérification manuelle

`python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['name'])"` affiche `eb-jepa-cifar10-comparison`.

#### Définition de terminé

- [ ] le TOML se charge.
- [ ] six dépendances runtime sont présentes.
- [ ] les packages sont cherchés sous src.

#### Indices

1. TOML utilise des sections entre crochets.
2. `requires-python` est distinct des dépendances.
3. Setuptools doit connaître la racine du layout.

---

### Exercice 0.3 — `Exports publics paresseux`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/__init__.py` et `src/comparison/cortical/__init__.py`.

#### Contexte

Les scripts importeront quelques symboles stables, mais plusieurs modules n'existent pas encore. Un import eager de tous les modules ferait échouer `import comparison` dès la phase 0.

#### Objectif

Définir `__all__` et résoudre les symboles seulement lorsqu'ils sont demandés.

#### Entrées

Un nom d'attribut tel que `ImageSSL` reçu par `__getattr__`.

#### Sorties

Le symbole importé ou une exception `AttributeError` pour un nom inconnu.

#### Comportement attendu

- déclarer les cinq noms publics du package principal.
- associer chaque nom à son module dans `__getattr__`.
- exposer seulement `FixedTreePredictor` dans le sous-package cortical.

#### Exemple

`import comparison` fonctionne maintenant ; `comparison.ImageSSL` ne sera résolu qu'une fois `comparison.model` créé.

#### Contraintes

- aucune fausse classe temporaire.
- un nom inconnu lève AttributeError.
- les imports de modules utilisent importlib ou des imports locaux.

#### Code de départ

```python
from typing import Any

__all__ = ["ImageSSL", "build_head", "build_model", "VICRegLoss", "load_config"]


def __getattr__(name: str) -> Any:
    ...
```

#### Étapes conseillées

1. écris la table nom vers module.
2. importe le module demandé dans la fonction.
3. retourne son attribut ou lève AttributeError.

#### Vérification manuelle

`PYTHONPATH=src python -c "import comparison; print(comparison.__all__)"` affiche les cinq noms sans importer PyTorch.

#### Définition de terminé

- [ ] le package s'importe.
- [ ] la liste publique est exacte.
- [ ] le sous-package n'expose que le prédicteur.

#### Indices

1. `__all__` documente l'API, il ne charge rien seul.
2. `__getattr__` au niveau module est appelé pour un attribut absent.
3. Une table évite une longue chaîne de conditions.

---

# Phase 1 — Copier les augmentations officielles

Toutes les classes de cette phase vivent dans `src/comparison/augmentation.py` et proviennent de `external/eb_jepa/examples/image_jepa/dataset.py:10-96`.

### Exercice 1.1 — `RandomResizedCrop`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/augmentation.py`. Copie `RandomResizedCrop` depuis la source officielle.

#### Contexte

Cette transformation prélève une zone aléatoire de l'image puis la redimensionne. Elle crée une vue différente sans changer la taille attendue par le backbone.

#### Objectif

Copier la classe officielle et conserver son tirage de paramètres et son interpolation.

#### Entrées

`image`, image PIL CIFAR-10 ; paramètres `size`, `scale` et `ratio`.

#### Sorties

Une image PIL carrée de taille `size × size`.

#### Comportement attendu

- stocker les paramètres dans `__init__`.
- tirer le rectangle avec l'utilitaire torchvision.
- recadrer et redimensionner dans `__call__`.

#### Exemple

Avec `size=32`, une image 32×32 produit toujours une vue 32×32 même si la zone choisie est plus petite.

#### Contraintes

- ne change pas la distribution aléatoire.
- conserve l'interpolation officielle.
- n'applique pas ToTensor ici.

#### Code de départ

```python
from PIL import Image

class RandomResizedCrop:
    def __init__(self, size: int, scale: tuple[float, float], ratio: tuple[float, float]) -> None: ...
    def __call__(self, image: Image.Image) -> Image.Image: ...
```

#### Étapes conseillées

1. retrouve la classe source.
2. copie ses attributs.
3. adapte seulement les annotations et imports.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import RandomResizedCrop; from PIL import Image; print(RandomResizedCrop(32,(.2,1.),(.75,1.33))(Image.new('RGB',(32,32))).size)"` affiche `(32, 32)`.

#### Définition de terminé

- [ ] sortie toujours 32×32.
- [ ] aucune conversion en tenseur.
- [ ] provenance ajoutée au fichier.

#### Indices

1. Le crop et le resize forment une seule opération.
2. Torchvision sait tirer les paramètres du crop.
3. Le corps officiel doit rester mathématiquement inchangé.

---

### Exercice 1.2 — `ColorJitter`

**Difficulté :** Facile

#### Où travailler

`src/comparison/augmentation.py`. Copie l'enveloppe `ColorJitter` officielle.

#### Contexte

Le jitter change luminosité, contraste, saturation et teinte seulement avec une probabilité donnée. L'enveloppe décide si la transformation torchvision est appelée.

#### Objectif

Copier le constructeur et l'appel probabiliste.

#### Entrées

`image` PIL et probabilité `p`.

#### Sorties

L'image transformée ou la même image si le tirage échoue.

#### Comportement attendu

- construire `transforms.ColorJitter` une fois.
- tirer un scalaire aléatoire à chaque appel.
- appliquer le jitter seulement si le scalaire est inférieur à p.

#### Exemple

`p=0` retourne l'image sans modification ; `p=1` exécute toujours le jitter.

#### Contraintes

- ne tire pas les paramètres dans __init__.
- n'enchaîne aucune autre augmentation.
- conserve les amplitudes officielles.

#### Code de départ

```python
class ColorJitter:
    def __init__(self, p: float = 0.8) -> None: ...
    def __call__(self, image): ...
```

#### Étapes conseillées

1. retrouve les quatre amplitudes.
2. stocke la transformation.
3. implémente le test de probabilité.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import ColorJitter; from PIL import Image; x=Image.new('RGB',(32,32)); print(ColorJitter(0)(x).size)"` affiche `(32, 32)`.

#### Définition de terminé

- [ ] p=0 et p=1 sont valides.
- [ ] la taille et le mode restent RGB.
- [ ] les amplitudes correspondent à la source.

#### Indices

1. L'enveloppe est distincte de torchvision.ColorJitter.
2. Le hasard doit être tiré à chaque vue.
3. Utilise le générateur PyTorch comme la source.

---

### Exercice 1.3 — `Grayscale`

**Difficulté :** Facile

#### Où travailler

`src/comparison/augmentation.py`. Copie `Grayscale`.

#### Contexte

Une vue peut perdre ses couleurs, mais le réseau attend toujours trois canaux.

#### Objectif

Copier la transformation probabiliste en niveaux de gris avec trois canaux de sortie.

#### Entrées

`image` PIL RGB et probabilité `p`.

#### Sorties

Image PIL RGB, grisée ou inchangée.

#### Comportement attendu

- tirer la décision à chaque appel.
- utiliser la transformation grayscale officielle.
- conserver trois canaux.

#### Exemple

Une image rouge grisée possède trois canaux égaux au lieu d'un seul canal.

#### Contraintes

- ne renvoie jamais une image mode L.
- probabilité officielle inchangée.
- aucune conversion Tensor.

#### Code de départ

```python
class Grayscale:
    def __init__(self, p: float = 0.2) -> None: ...
    def __call__(self, image): ...
```

#### Étapes conseillées

1. construis l'opérateur grayscale.
2. fixe num_output_channels à 3.
3. entoure l'appel du test probabiliste.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import Grayscale; from PIL import Image; print(Grayscale(1)(Image.new('RGB',(4,4),'red')).mode)"` affiche `RGB`.

#### Définition de terminé

- [ ] sortie RGB.
- [ ] p respectée.
- [ ] aucune modification en dehors de la classe.

#### Indices

1. Le backbone exige trois canaux.
2. Torchvision peut dupliquer le canal gris.
3. La source indique directement le nombre de canaux.

---

### Exercice 1.4 — `Solarization`

**Difficulté :** Facile

#### Où travailler

`src/comparison/augmentation.py`. Copie `Solarization`.

#### Contexte

La solarisation inverse les intensités au-dessus d'un seuil et crée une perturbation photométrique forte.

#### Objectif

Copier l'enveloppe probabiliste avec seuil 128.

#### Entrées

`image` PIL et probabilité `p`.

#### Sorties

Image solarisée ou inchangée.

#### Comportement attendu

- tirer une décision.
- appeler `ImageOps.solarize` avec 128.
- retourner l'objet obtenu.

#### Exemple

Un pixel de valeur 200 devient 55 lorsque la solarisation est appliquée.

#### Contraintes

- seuil exactement 128.
- pas de normalisation préalable.
- probabilité conforme à la source.

#### Code de départ

```python
class Solarization:
    def __init__(self, p: float = 0.0) -> None: ...
    def __call__(self, image): ...
```

#### Étapes conseillées

1. importe PIL.ImageOps.
2. stocke p.
3. applique solarize dans la branche vraie.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import Solarization; from PIL import Image; print(Solarization(1)(Image.new('RGB',(1,1),(200,200,200))).getpixel((0,0)))"` affiche `(55, 55, 55)`.

#### Définition de terminé

- [ ] seuil vérifié.
- [ ] p=0 laisse l'image intacte.
- [ ] mode et taille conservés.

#### Indices

1. La formule PIL au-dessus du seuil est 255-x.
2. Le seuil est un argument de solarize.
3. Ne confonds pas solarisation et inversion complète.

---

### Exercice 1.5 — `HorizontalFlip`

**Difficulté :** Facile

#### Où travailler

`src/comparison/augmentation.py`. Copie `HorizontalFlip`.

#### Contexte

Le retournement horizontal modifie la géométrie sans changer la classe sémantique d'une image CIFAR-10.

#### Objectif

Copier le flip probabiliste officiel.

#### Entrées

`image` PIL ; probabilité fixée à 0,5.

#### Sorties

Image retournée ou inchangée, même taille.

#### Comportement attendu

- tirer la décision à chaque appel.
- utiliser le flip horizontal fonctionnel.
- retourner l'image.

#### Exemple

Sur une image dont la colonne gauche est blanche, un flip réussi déplace cette colonne à droite.

#### Contraintes

- p=0,5.
- jamais de flip vertical.
- pas de mutation en place exigée.

#### Code de départ

```python
class HorizontalFlip:
    def __init__(self, p: float = 0.5) -> None: ...
    def __call__(self, image): ...
```

#### Étapes conseillées

1. stocke p.
2. tire un nombre aléatoire.
3. appelle la fonction horizontale dans la branche vraie.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import HorizontalFlip; print(HorizontalFlip().p)"` affiche `0.5`.

#### Définition de terminé

- [ ] probabilité exacte.
- [ ] taille inchangée.
- [ ] fonction horizontale uniquement.

#### Indices

1. Le flip se décide par vue.
2. Torchvision fournit une fonction adaptée aux images PIL.
3. Une propriété p rend le comportement lisible.

---

### Exercice 1.6 — `get_train_transforms`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/augmentation.py`. Copie et adapte la fabrique officielle.

#### Contexte

Les cinq transformations précédentes ne suffisent pas seules : elles doivent être composées dans le même ordre, puis l'image convertie et normalisée pour ResNet.

#### Objectif

Construire le pipeline complet CIFAR-10 et retourner un callable.

#### Entrées

`crop_scale`, intervalle de surfaces relatives ; options de jitter, blur, solarisation et flip.

#### Sorties

Callable transformant une image PIL en Tensor `(3,32,32)` normalisé.

#### Comportement attendu

- crop redimensionné.
- flip, jitter, grayscale, blur et solarisation dans l'ordre officiel.
- ToTensor puis normalisation avec les statistiques officielles.

#### Exemple

Deux appels sur la même image peuvent produire deux tenseurs différents, mais toujours de forme `(3,32,32)`.

#### Contraintes

- taille de sortie 32.
- ordre officiel conservé.
- aucune augmentation dépendante du label.

#### Code de départ

```python
def get_train_transforms(crop_scale: tuple[float, float] = (0.2, 1.0)):
    """Construit le pipeline stochastique officiel adapté à CIFAR-10."""
    ...
```

#### Étapes conseillées

1. liste les opérateurs dans leur ordre source.
2. adapte seulement la taille du crop à 32.
3. termine par tensor puis normalize.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.augmentation import get_train_transforms; from PIL import Image; print(get_train_transforms()(Image.new('RGB',(32,32))).shape)"` affiche `torch.Size([3, 32, 32])`.

#### Définition de terminé

- [ ] forme correcte.
- [ ] valeurs finies.
- [ ] deux appels sont indépendants.
- [ ] ordre identique à la référence.

#### Indices

1. Compose exécute sa liste de gauche à droite.
2. Le blur travaille encore sur l'image avant normalisation.
3. La normalisation doit être la dernière opération.

---

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

---

# Phase 3 — Copier le backbone et le projecteur baseline

Cette phase crée la branche de référence. Le backbone produit 512 features ; la tête MLP les projette vers 2048 dimensions pour VICReg.

### Exercice 3.1 — `ResNet18`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/backbone.py`. Source : `external/eb_jepa/examples/image_jepa/main.py:61-75`.

#### Contexte

La ResNet torchvision standard est conçue pour de grandes images. La référence EB-JEPA remplace sa couche finale et adapte son stem aux images CIFAR-10 32×32.

#### Objectif

Copier la classe officielle sans poids préentraînés et exposer une sortie `(B,512)`.

#### Entrées

`images: Tensor` de forme `(B,3,32,32)`.

#### Sorties

Features `Tensor` de forme `(B,512)`.

#### Comportement attendu

- construire `torchvision.models.resnet18(weights=None)`.
- remplacer `fc` par `nn.Identity`.
- remplacer conv1 par une convolution 3×3 stride 1 padding 2.
- remplacer maxpool par Identity.
- déléguer forward au modèle.

#### Exemple

Pour huit images, `ResNet18()(images)` donne `(8,512)` ; aucune classification en dix classes n'est calculée.

#### Contraintes

- `features_dim = 512`.
- aucun téléchargement de poids.
- une seule passe du backbone.
- copie attribuée.

#### Code de départ

```python
class ResNet18(nn.Module):
    features_dim: int = 512

    def __init__(self) -> None:
        super().__init__()
        ...

    def forward(self, images: Tensor) -> Tensor:
        ...
```

#### Étapes conseillées

1. retrouve le constructeur source.
2. inspecte les modules conv1, maxpool et fc.
3. remplace seulement ces trois points.
4. écris le forward.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.backbone import ResNet18; print(ResNet18()(torch.randn(2,3,32,32)).shape)"` affiche `[2, 512]`.

#### Définition de terminé

- [ ] conv1 3×3/stride1/padding2.
- [ ] maxpool et fc sont Identity.
- [ ] sortie `(2,512)` finie.

#### Indices

1. La classification réside dans `fc`.
2. Garder le max-pool réduirait trop vite la carte 32×32.
3. La classe peut stocker la ResNet dans `self.model` puis déléguer.

---

### Exercice 3.2 — `MLPProjector`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/heads.py`. Source : bloc `ImageSSL.projector` dans `external/eb_jepa/examples/image_jepa/main.py:78-102`.

#### Contexte

Le baseline doit être le projecteur officiel, pas un MLP simplifié. BatchNorm apparaît entre les Linear et ReLU, jamais après la dernière Linear.

#### Objectif

Transformer le `nn.Sequential` officiel en classe nommée sans changer l'ordre ou les dimensions.

#### Entrées

`features: Tensor` `(B,512)` ; dimensions configurables 512/2048/2048.

#### Sorties

Projections `Tensor` `(B,2048)`.

#### Comportement attendu

- construire Linear 512→2048.
- ajouter BatchNorm1d puis ReLU.
- répéter Linear 2048→2048, BatchNorm, ReLU.
- terminer par Linear 2048→2048.
- déléguer forward au Sequential.

#### Exemple

Un batch de huit features traverse neuf modules et ressort en `(8,2048)`, sans activation finale.

#### Contraintes

- trois Linear exactement.
- deux BatchNorm et deux ReLU.
- aucune branche liée à la tête corticale.
- pas d'activation finale.

#### Code de départ

```python
class MLPProjector(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 2048) -> None:
        super().__init__()
        ...

    def forward(self, features: Tensor) -> Tensor:
        ...
```

#### Étapes conseillées

1. copie l'ordre du Sequential source.
2. remplace les constantes par les trois arguments.
3. stocke le bloc.
4. écris le forward.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.heads import MLPProjector; m=MLPProjector().eval(); print(m(torch.randn(8,512)).shape)"` affiche `[8, 2048]`.

#### Définition de terminé

- [ ] architecture 3 Linear/2 BN/2 ReLU.
- [ ] sortie `(8,2048)`.
- [ ] paramètres entraînables.
- [ ] provenance documentée.

#### Indices

1. Compte les couches dans la source.
2. BatchNorm1d attend `(B,D)`.
3. `self.layers = nn.Sequential(...)` suffit ; forward retourne `self.layers(features)`.

---

# Phase 4 — Copier et raccorder le prédicteur cortical

Les exercices 4.1 à 4.6 sont des copies guidées depuis `Predictor/`. Les exercices 4.7 et 4.8 raccordent ce code au projet de comparaison.

### Exercice 4.1 — `Arbre et nœuds`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/cortical/tree_structure.py`. Source : `Predictor/utils/tree_structure.py`.

#### Contexte

Le prédicteur possède un arbre n-aire fixe numéroté en largeur. Toutes les colonnes utilisent ces relations parent/enfants pour transmettre leurs états.

#### Objectif

Copier `ColumnNode`, `num_columns`, `build_tree_positions`, `level_of`, `children_of`, `parent_of` et `is_leaf`.

#### Entrées

`n` facteur de branchement ≥2 ; `max_depth` profondeur ≥0 ; un `ColumnNode`.

#### Sorties

Nombre de colonnes, liste BFS et relations structurelles.

#### Comportement attendu

- retirer seulement l'import pdb.
- conserver validations et ordre BFS.
- conserver identifiants, niveaux et relations parentales.

#### Exemple

`n=2,max_depth=2` crée 7 nœuds : racine, deux nœuds niveau 1, quatre feuilles niveau 2.

#### Contraintes

- aucune nouvelle dépendance.
- n<2 et profondeur négative refusés.
- identifiants uniques.
- équations inchangées.

#### Code de départ

```python
from dataclasses import dataclass

@dataclass
class ColumnNode:
    ...

def num_columns(n: int, max_depth: int) -> int: ...
def build_tree_positions(n: int, max_depth: int) -> list[ColumnNode]: ...
def level_of(node: ColumnNode) -> int: ...
def children_of(node: ColumnNode) -> list[ColumnNode]: ...
def parent_of(node: ColumnNode) -> ColumnNode | None: ...
def is_leaf(node: ColumnNode, max_depth: int) -> bool: ...
```

#### Étapes conseillées

1. ouvre la source.
2. copie les sept symboles.
3. supprime pdb.
4. ajoute le commentaire de provenance.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.cortical.tree_structure import *; print(num_columns(2,2), len(build_tree_positions(2,2)))"` affiche `7 7`.

#### Définition de terminé

- [ ] sept symboles présents.
- [ ] ordre BFS.
- [ ] relations cohérentes.
- [ ] erreurs de bornes conservées.

#### Indices

1. La somme géométrique donne le nombre de nœuds.
2. BFS place tous les nœuds d'un niveau ensemble.
3. Ne réécris pas la structure : copie-la.

---

### Exercice 4.2 — `FiLMModulation`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/cortical/modulation.py`. Source : `Predictor/modules/modulation.py`.

#### Contexte

FiLM transforme un contexte en deux vecteurs, gamma et beta, qui modulent un état caché normalisé.

#### Objectif

Copier la classe et conserver exactement `gamma * normalized_hidden + beta`.

#### Entrées

`hidden (B,dim_h)` et `context (B,dim_c)`.

#### Sorties

Tensor modulé `(B,dim_h)`.

#### Comportement attendu

- LayerNorm de hidden sans affine.
- projection du contexte vers 2×dim_h.
- séparation gamma/beta.
- combinaison FiLM.

#### Exemple

Avec `hidden (4,256)` et `context (4,128)`, la sortie reste `(4,256)`.

#### Contraintes

- LayerNorm `elementwise_affine=False`.
- aucune mutation d'entrée.
- équation inchangée.

#### Code de départ

```python
class FiLMModulation(nn.Module):
    def __init__(self, dim_h: int, dim_c: int) -> None: ...
    def forward(self, hidden: Tensor, context: Tensor) -> Tensor: ...
```

#### Étapes conseillées

1. copie les imports utiles.
2. copie le constructeur.
3. copie forward.
4. documente la source.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.cortical.modulation import FiLMModulation as F; print(F(8,3)(torch.randn(2,8),torch.randn(2,3)).shape)"` affiche `[2,8]`.

#### Définition de terminé

- [ ] forme conservée.
- [ ] LayerNorm non affine.
- [ ] gamma/beta issus du contexte.

#### Indices

1. La projection produit deux fois dim_h.
2. `chunk(2, dim=-1)` sépare les deux parties.
3. Le hidden normalisé est multiplié avant l'ajout.

---

### Exercice 4.3 — `Colonnes corticales`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/cortical/column.py`. Source : `Predictor/modules/column.py`.

#### Contexte

Une colonne encode son signal, peut agréger les états de ses sœurs et recevoir un feedback. C'est le cœur récurrent du prédicteur.

#### Objectif

Copier `_lateral_attention`, `ColumnStep` et `CorticalColumn` en corrigeant uniquement l'annotation `torch.torch.Tensor`.

#### Entrées

État propre `(B,H)`, liste d'états sœurs `(B,H)`, feedback éventuel et entrée de colonne.

#### Sorties

Nouvel état caché de colonne `(B,H)`.

#### Comportement attendu

- copier les projections query/key/value.
- conserver softmax sur les sœurs.
- conserver résidu, modulation et ordre des opérations.
- corriger l'annotation en `torch.Tensor`.

#### Exemple

Sans sœur, l'attention latérale contribue un zéro compatible ; avec deux sœurs elle calcule deux poids par élément du batch.

#### Contraintes

- aucune équation modifiée.
- poids appartenant à la colonne receveuse.
- flags lateral/feedback conservés.
- pas de pdb.

#### Code de départ

```python
def _lateral_attention(W_q, W_k, W_v, H_self: Tensor, H_others: list[Tensor]) -> Tensor: ...

class ColumnStep(nn.Module): ...
class CorticalColumn(nn.Module): ...
```

#### Étapes conseillées

1. copie d'abord le helper.
2. copie ColumnStep et ses modules.
3. copie l'enveloppe CorticalColumn.
4. répare uniquement l'annotation.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.cortical.column import ColumnStep, CorticalColumn; print(ColumnStep, CorticalColumn)"` doit importer les deux classes.

#### Définition de terminé

- [ ] import réussi.
- [ ] annotation valide.
- [ ] cas sans sœur géré.
- [ ] équations source intactes.

#### Indices

1. Une liste vide demande un cas explicite.
2. L'attention porte sur les autres colonnes du même niveau.
3. Ce travail est une copie guidée, pas un refactoring.

---

### Exercice 4.4 — `Décomposition et feedback`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/cortical/decomposition.py`. Source : `Predictor/modules/decomposition.py`.

#### Contexte

Un parent doit produire une entrée distincte pour chaque enfant ; le feedback redescend lui aussi par des projections propres.

#### Objectif

Copier `ChildDecomposition` et `FeedbackProjection`.

#### Entrées

État parent `(B,dim_parent)` et facteur `n`.

#### Sorties

Liste de `n` tenseurs, un par enfant.

#### Comportement attendu

- créer un ModuleList de n Linear.
- appliquer chaque Linear au même état parent.
- retourner une liste dans l'ordre des enfants.

#### Exemple

`n=2` produit deux tenseurs de même forme mais issus de deux couches Linear différentes.

#### Contraintes

- aucun partage de couche entre enfants.
- longueur de liste exactement n.
- ordre stable.

#### Code de départ

```python
class ChildDecomposition(nn.Module): ...
class FeedbackProjection(nn.Module): ...
```

#### Étapes conseillées

1. copie la première classe.
2. vérifie son ModuleList.
3. copie la seconde.
4. adapte les imports.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.cortical.decomposition import ChildDecomposition, FeedbackProjection; print(ChildDecomposition, FeedbackProjection)"` doit réussir.

#### Définition de terminé

- [ ] deux classes importables.
- [ ] n projections distinctes.
- [ ] formes conformes.

#### Indices

1. ModuleList enregistre les paramètres.
2. Multiplier une même référence Linear partagerait les poids.
3. Utilise une compréhension qui construit chaque Linear.

---

### Exercice 4.5 — `Intégration récursive`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/cortical/integration.py`. Source : `Predictor/modules/integration.py`.

#### Contexte

Après propagation descendante, les feuilles produisent des vecteurs U. Les parents concatènent les U de leurs enfants et les combinent à leur propre état jusqu'à la racine.

#### Objectif

Copier `LeafProjection`, `RecursiveIntegration` et `LatentOutputProjection`.

#### Entrées

État de feuille/parent `(B,H)` ; liste de n enfants `(B,U)`.

#### Sorties

Vecteur intégré `(B,U)`, puis sortie cible `(B,2048)`.

#### Comportement attendu

- projeter une feuille H→U.
- concaténer les enfants sur la dernière dimension.
- combiner agrégat et état parent.
- projeter U vers dim_target.

#### Exemple

Deux enfants `(4,256)` deviennent `(4,512)` avant l'intégration ; la racine finit en `(4,2048)`.

#### Contraintes

- concaténation sur dim=-1.
- ordre des enfants conservé.
- aucune moyenne à la place de concat.
- équations source intactes.

#### Code de départ

```python
class LeafProjection(nn.Module): ...
class RecursiveIntegration(nn.Module): ...
class LatentOutputProjection(nn.Module): ...
```

#### Étapes conseillées

1. copie les trois classes.
2. vérifie leurs dimensions constructeur.
3. adapte les imports.
4. documente la provenance.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.cortical.integration import LeafProjection, RecursiveIntegration, LatentOutputProjection; print('ok')"` affiche `ok`.

#### Définition de terminé

- [ ] trois classes présentes.
- [ ] concaténation correcte.
- [ ] sortie target configurable.

#### Indices

1. Une feuille n'a pas d'enfants.
2. Le parent reçoit n×dim_U après concaténation.
3. La dernière projection est la seule qui vise 2048.

---

### Exercice 4.6 — `FixedTreePredictor`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/cortical/fixed_tree_predictor.py`. Source : `Predictor/model/fixed_tree_predictor.py`.

#### Contexte

Cette classe assemble arbre, colonnes, décomposition, feedback et intégration. Elle doit rester comportementalement identique au prédicteur local d'origine.

#### Objectif

Copier la classe complète, remplacer ses imports par des imports relatifs et retirer sa démonstration exécutable.

#### Entrées

Features `(B,dim_in)` et paramètres structurels n/L_max/dimensions.

#### Sorties

Représentation `(B,dim_target)`.

#### Comportement attendu

- construire tous les modules selon l'arbre.
- propager les états du haut vers le bas.
- respecter disable_lateral et disable_feedback.
- intégrer les feuilles vers la racine.
- appliquer la projection de sortie.

#### Exemple

`n=2,L_max=2` instancie sept colonnes ; une entrée `(4,512)` produit `(4,2048)`.

#### Contraintes

- imports relatifs `.column`, `.integration`, etc..
- aucun bloc main.
- aucune modification des équations.
- un encode par colonne.

#### Code de départ

```python
class FixedTreePredictor(nn.Module):
    def __init__(self, n: int, L_max: int, dim_in: int, dim_hidden: int,
                 dim_feedback: int, dim_U: int, dim_target: int,
                 disable_lateral: bool = False, disable_feedback: bool = False) -> None: ...
    def forward(self, features: Tensor) -> Tensor: ...
```

#### Étapes conseillées

1. copie le fichier source.
2. convertis uniquement les imports.
3. retire le bloc de démonstration.
4. compare les state_dict.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.cortical.fixed_tree_predictor import FixedTreePredictor as F; m=F(2,2,512,256,128,256,2048); print(m(torch.randn(2,512)).shape)"` affiche `[2,2048]`.

#### Définition de terminé

- [ ] sept colonnes pour 2/2.
- [ ] forme `(2,2048)`.
- [ ] flags utilisables.
- [ ] imports autonomes.

#### Indices

1. Commence par une copie exacte.
2. Les imports sont la seule adaptation structurelle.
3. Un chargement strict du state_dict source permet de contrôler l'équivalence.

---

### Exercice 4.7 — `CorticalHead`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/heads.py`, après `MLPProjector`. Importe `FixedTreePredictor` depuis `.cortical`.

#### Contexte

Le modèle commun attend une tête callable 512→2048. Le prédicteur satisfait presque ce contrat, mais un adaptateur doit détecter immédiatement une mauvaise dimension sans ajouter de paramètres.

#### Objectif

Écrire un `nn.Module` qui valide l'entrée, appelle le prédicteur une fois, valide la sortie et la retourne.

#### Entrées

`predictor`, instance déjà construite ; `features: Tensor (B,512)` ; dimensions attendues.

#### Sorties

Le Tensor exact retourné par predictor, de forme `(B,2048)`.

#### Comportement attendu

- stocker predictor et dimensions.
- refuser un Tensor dont la dernière dimension n'est pas input_dim.
- appeler predictor(features) exactement une fois.
- refuser une mauvaise dimension de sortie.
- retourner projections sans nouvelle couche.

#### Exemple

Avec un faux predictor retournant des zéros `(B,2048)`, l'adaptateur accepte `(3,512)` ; il refuse `(3,511)` avec un message donnant 512 et 511.

#### Contraintes

- aucun Linear, BatchNorm ou paramètre supplémentaire.
- ne détache pas le gradient.
- validation sur la dernière dimension.
- message d'erreur explicite.

#### Code de départ

```python
from .cortical import FixedTreePredictor

class CorticalHead(nn.Module):
    def __init__(self, predictor: FixedTreePredictor, input_dim: int = 512, output_dim: int = 2048) -> None:
        super().__init__()
        ...

    def forward(self, features: Tensor) -> Tensor:
        ...
```

#### Étapes conseillées

1. stocke les trois arguments utiles.
2. vérifie features.ndim et features.shape[-1].
3. appelle self.predictor.
4. vérifie la sortie.
5. retourne-la.

#### Vérification manuelle

`PYTHONPATH=src python - <<'PY'
import torch
from comparison.heads import CorticalHead
class P(torch.nn.Module):
    def forward(self,x): return torch.zeros(x.shape[0],2048)
print(CorticalHead(P())(torch.randn(3,512)).shape)
PY` affiche `[3,2048]`.

#### Définition de terminé

- [ ] aucun paramètre propre hors predictor.
- [ ] entrée/sortie validées.
- [ ] gradient conservé.
- [ ] predictor appelé une fois.

#### Indices

1. Une validation de forme ne nécessite pas de couche.
2. Les paramètres du predictor apparaîtront déjà via le sous-module.
3. Compare `shape[-1]`, pas la taille du batch.

---

### Exercice 4.8 — `build_head`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/heads.py`, après les deux classes de tête. Types de configuration définis en phase 8 mais accessibles par clés OmegaConf.

#### Contexte

Le reste du projet ne doit jamais connaître le choix baseline/cortical. Cette factory est l'unique branche architecturale.

#### Objectif

Construire `MLPProjector` pour `baseline` ou un `FixedTreePredictor` enveloppé par `CorticalHead` pour `cortical`.

#### Entrées

`cfg: DictConfig` contenant `model.head_type`, dimensions baseline et section cortical.

#### Sorties

Un nouveau `nn.Module` de contrat `(B,512)→(B,2048)`.

#### Comportement attendu

- lire head_type une seule fois.
- pour baseline, passer les dimensions model au MLP.
- pour cortical, passer chaque champ cortical au predictor puis l'envelopper.
- lever ValueError pour toute autre chaîne.

#### Exemple

Deux appels avec la même config créent deux objets sans paramètres partagés. `head_type='foo'` échoue immédiatement.

#### Contraintes

- seul branchement architectural du projet.
- aucun cache d'instance.
- dimensions 512/2048 tirées de la config.
- type inconnu refusé.

#### Code de départ

```python
def build_head(cfg: DictConfig) -> nn.Module:
    """Construit une nouvelle tête baseline ou corticale."""
    ...
```

#### Étapes conseillées

1. lis et normalise head_type.
2. écris la branche baseline.
3. écris la branche cortical avec tous les arguments.
4. ajoute l'erreur finale.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.heads import build_head; from omegaconf import OmegaConf; c=OmegaConf.create({'model':{'head_type':'unknown'}}); build_head(c)"` doit lever `ValueError` mentionnant `unknown`.

#### Définition de terminé

- [ ] deux types acceptés.
- [ ] type inconnu rejeté.
- [ ] aucun partage entre appels.
- [ ] aucune branche ailleurs requise.

#### Indices

1. OmegaConf autorise `cfg.model.head_type`.
2. La branche cortical construit d'abord le predictor.
3. Termine par `raise ValueError(...)` plutôt que retourner None.

---

# Phase 5 — Construire le modèle commun

Crée `src/comparison/model.py`. À ce stade `ResNet18`, `MLPProjector`, `CorticalHead` et `build_head` existent déjà.

### Exercice 5.1 — `ImageSSL`

**Difficulté :** Facile

#### Où travailler

Crée `src/comparison/model.py`. Source conceptuelle : `external/eb_jepa/examples/image_jepa/main.py:78-102`.

#### Contexte

Dans la classe officielle, le backbone et le projecteur sont construits ensemble. Ici, ils sont injectés afin que le même conteneur accepte soit le MLP, soit l'arbre cortical. Le modèle retourne aussi les features pour les diagnostics.

#### Objectif

Écrire le conteneur qui applique d'abord le backbone aux images, puis la tête aux features, et retourne les deux résultats.

#### Entrées

`backbone: ResNet18` transforme `(B,3,32,32)` en `(B,512)` ; `head: nn.Module` transforme `(B,512)` en `(B,2048)` ; `images` est le batch reçu par forward.

#### Sorties

Tuple `(features, projections)` avec formes `(B,512)` et `(B,2048)`.

#### Comportement attendu

- dans __init__, enregistrer backbone et head comme sous-modules.
- dans forward, appeler `self.backbone(images)` une seule fois.
- passer exactement ce résultat à `self.head`.
- retourner `(features, projections)` dans cet ordre.

#### Exemple

Pour `images=torch.randn(8,3,32,32)`, le premier élément du tuple est `(8,512)` et le second `(8,2048)`. VICReg consommera le second ; le diagnostic pourra observer le premier.

#### Contraintes

- aucune création de couche dans cette classe.
- aucune condition baseline/cortical.
- ne pas détacher les features.
- conserver le tuple officiel.

#### Code de départ

```python
from torch import Tensor, nn

from .backbone import ResNet18


class ImageSSL(nn.Module):
    def __init__(self, backbone: ResNet18, head: nn.Module) -> None:
        super().__init__()
        # Enregistre ici les deux modules reçus.
        ...

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        # 1. Calcule les features.
        # 2. Calcule les projections depuis ces mêmes features.
        # 3. Retourne les deux dans cet ordre.
        ...
```

#### Étapes conseillées

1. crée le fichier avec les imports montrés.
2. assigne les deux arguments à self.
3. écris les deux appels dans forward.
4. retourne le tuple.

#### Vérification manuelle

`PYTHONPATH=src python - <<'PY'
import torch
from comparison.model import ImageSSL
backbone=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,512))
head=torch.nn.Linear(512,2048)
f,z=ImageSSL(backbone,head)(torch.randn(4,3,32,32))
print(f.shape,z.shape)
PY` affiche `(4,512) (4,2048)`.

#### Définition de terminé

- [ ] backbone appelé une fois.
- [ ] head reçoit les features.
- [ ] formes exactes.
- [ ] aucune branche architecturale.

#### Indices

1. Un nn.Module reçu doit être assigné à `self` pour enregistrer ses paramètres.
2. La tête ne reçoit jamais directement les images.
3. Le corps de forward tient en trois lignes : calcul, calcul, retour.

---

### Exercice 5.2 — `build_model`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/model.py`, sous `ImageSSL`. Importe `build_head` depuis `.heads`.

#### Contexte

La factory garantit qu'un nouveau backbone est construit pour chaque run. La seule différence entre les deux modèles vient de `build_head(cfg)`.

#### Objectif

Assembler un `ResNet18` neuf et la tête demandée dans `ImageSSL`.

#### Entrées

`cfg: DictConfig`, configuration complète validée.

#### Sorties

Nouvelle instance `ImageSSL`.

#### Comportement attendu

- instancier ResNet18 sans argument caché.
- appeler build_head exactement une fois.
- passer les deux objets à ImageSSL.
- retourner le modèle.

#### Exemple

Deux configurations identiques sauf `head_type` produisent deux ResNet-18 de même structure, mais deux têtes différentes et aucun objet Parameter partagé.

#### Contraintes

- aucun chargement de checkpoint ici.
- aucune seed modifiée.
- aucune condition head_type.
- nouvelle instance à chaque appel.

#### Code de départ

```python
def build_model(cfg: DictConfig) -> ImageSSL:
    ...
```

#### Étapes conseillées

1. importe DictConfig, ResNet18 et build_head.
2. construis le backbone.
3. construis la tête.
4. retourne ImageSSL.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.model import build_model; print(build_model.__annotations__)"` doit annoncer un retour `ImageSSL`.

#### Définition de terminé

- [ ] factory importable.
- [ ] aucune branche sur head_type.
- [ ] instances neuves.
- [ ] composition correcte.

#### Indices

1. Le branchement existe déjà dans build_head.
2. Une factory ne lance aucun forward.
3. Trois affectations suffisent.

---

### Exercice 5.3 — `ParameterCounts`

**Difficulté :** Facile

#### Où travailler

`src/comparison/model.py`, sous `build_model`.

#### Contexte

Le projet cherche une réduction de compute. Le nombre de paramètres entraînables est un premier indicateur, à séparer entre backbone et tête.

#### Objectif

Créer une dataclass immuable et une fonction qui somme les `numel()` des paramètres dont `requires_grad` est vrai.

#### Entrées

`model: ImageSSL`.

#### Sorties

`ParameterCounts(backbone=int, head=int, total=int)`.

#### Comportement attendu

- écrire un helper de somme sur un module.
- compter séparément model.backbone et model.head.
- calculer total comme somme des deux.
- retourner des int Python.

#### Exemple

Si le backbone compte 11M et la tête 9M, le résultat contient 11M, 9M et 20M ; un paramètre gelé n'est pas compté.

#### Contraintes

- uniquement requires_grad=True.
- pas de double comptage.
- total égal à backbone+head.
- dataclass frozen.

#### Code de départ

```python
@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    head: int
    total: int


def count_parameters(model: ImageSSL) -> ParameterCounts:
    ...
```

#### Étapes conseillées

1. écris la somme générique.
2. applique-la aux deux sous-modules.
3. additionne.
4. construis la dataclass.

#### Vérification manuelle

`PYTHONPATH=src python - <<'PY'
import torch
from comparison.model import ImageSSL,count_parameters
m=ImageSSL(torch.nn.Linear(2,3),torch.nn.Linear(3,4))
print(count_parameters(m))
PY` doit donner backbone 9, head 16, total 25.

#### Définition de terminé

- [ ] comptes exacts.
- [ ] paramètres gelés exclus.
- [ ] total cohérent.

#### Indices

1. Un Linear a poids plus biais.
2. `p.numel()` retourne un entier.
3. Le modèle ne contient que backbone et head.

---

# Phase 6 — Copier VICReg et diagnostiquer l'effondrement

Écris `src/comparison/losses.py`. Les trois premières classes proviennent de `external/eb_jepa/eb_jepa/losses.py` au commit figé.

### Exercice 6.1 — `HingeStdLoss`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/losses.py`. Source : `eb_jepa/losses.py:56-81`.

#### Contexte

Le terme de variance empêche toutes les images d'obtenir le même vecteur. Il pénalise chaque dimension dont l'écart-type est inférieur à une cible.

#### Objectif

Copier la classe officielle, y compris centrage, variance, epsilon et hinge.

#### Entrées

Représentation `x (B,D)`, target=1, epsilon=1e-4.

#### Sorties

Scalaire différentiable moyen sur les D dimensions.

#### Comportement attendu

- centrer x sur le batch.
- calculer variance par dimension.
- prendre sqrt(variance+epsilon).
- appliquer relu(target-std).
- moyenner.

#### Exemple

Si toutes les lignes sont identiques, std≈0 et la pénalité approche 1 ; si chaque dimension varie assez, elle approche 0.

#### Contraintes

- dimension batch correcte.
- epsilon sous la racine.
- gradient conservé.
- copie mathématique exacte.

#### Code de départ

```python
class HingeStdLoss(nn.Module):
    def __init__(self, target: float = 1.0, epsilon: float = 1e-4) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

#### Étapes conseillées

1. copie la classe source.
2. conserve l'ordre numérique.
3. adapte imports/types.
4. ajoute provenance.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.losses import HingeStdLoss; print(float(HingeStdLoss()(torch.zeros(8,4))))"` affiche une valeur proche de 0.99.

#### Définition de terminé

- [ ] sortie scalaire.
- [ ] effondrement pénalisé.
- [ ] epsilon exact.
- [ ] gradient possible.

#### Indices

1. La variance est calculée colonne par colonne.
2. Le hinge est une ReLU.
3. Ne convertis pas le scalaire avec item() dans forward.

---

### Exercice 6.2 — `CovarianceLoss`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/losses.py`. Source : `eb_jepa/losses.py:84-111`.

#### Contexte

Une représentation peut varier tout en dupliquant la même information dans toutes ses dimensions. Le terme covariance pénalise seulement les corrélations hors diagonale.

#### Objectif

Copier la covariance officielle et sa méthode de retrait de diagonale.

#### Entrées

`x (B,D)` avec B>1.

#### Sorties

Scalaire différentiable représentant l'énergie hors diagonale.

#### Comportement attendu

- centrer le batch.
- calculer covariance D×D avec normalisation B-1.
- extraire les éléments hors diagonale.
- élever au carré, sommer et normaliser.

#### Exemple

Pour des dimensions identiques, les covariances hors diagonale sont grandes ; pour des dimensions décorrélées elles diminuent.

#### Contraintes

- ne pénalise pas la diagonale.
- division B-1.
- méthode off_diagonal copiée.
- pas de boucle Python sur D.

#### Code de départ

```python
class CovarianceLoss(nn.Module):
    @staticmethod
    def off_diagonal(x: Tensor) -> Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

#### Étapes conseillées

1. copie le helper.
2. vérifie que la matrice est carrée.
3. copie le centrage et produit matriciel.
4. conserve la normalisation.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.losses import CovarianceLoss; print(CovarianceLoss()(torch.randn(8,4)).shape)"` affiche `torch.Size([])`.

#### Définition de terminé

- [ ] diagonale exclue.
- [ ] scalaire fini.
- [ ] batch B>1 documenté.

#### Indices

1. La covariance est `x.T @ x`.
2. Le helper officiel aplatit sans créer de masque dense.
3. La sortie doit rester dans le graphe autograd.

---

### Exercice 6.3 — `VICRegLoss`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/losses.py`. Source : `eb_jepa/losses.py:306-342`.

#### Contexte

VICReg combine invariance entre deux vues, variance sur chaque vue et covariance sur chaque vue. La même instance sert aux deux architectures.

#### Objectif

Copier la classe et conserver les coefficients 1, 1 et 80 du protocole.

#### Entrées

`z1,z2 (B,2048)` ; coefficients et paramètres variance.

#### Sorties

Scalaire total différentiable ; expose aussi les composantes si la source les conserve.

#### Comportement attendu

- calculer MSE d'invariance entre z1 et z2.
- appliquer HingeStdLoss séparément aux deux vues puis moyenner.
- appliquer CovarianceLoss séparément puis moyenner.
- combiner avec coefficients.

#### Exemple

Si z1=z2, l'invariance vaut zéro, mais une représentation constante garde une forte pénalité de variance : égalité seule ne suffit pas.

#### Contraintes

- coefficients configurables mais identiques entre têtes.
- deux vues de même forme.
- aucun label.
- pondération officielle exacte.

#### Code de départ

```python
class VICRegLoss(nn.Module):
    def __init__(self, invariance_coeff: float = 1.0, std_coeff: float = 1.0,
                 cov_coeff: float = 80.0, variance_target: float = 1.0,
                 variance_epsilon: float = 1e-4) -> None: ...
    def forward(self, z1: Tensor, z2: Tensor) -> Tensor: ...
```

#### Étapes conseillées

1. copie le constructeur source.
2. instancie les deux pertes auxiliaires.
3. copie les trois calculs.
4. combine sans item().

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.losses import VICRegLoss; z=torch.randn(8,2048,requires_grad=True); l=VICRegLoss()(z,z); l.backward(); print(l.shape,z.grad is not None)"` affiche `[] True`.

#### Définition de terminé

- [ ] backward fonctionne.
- [ ] aucun label.
- [ ] coefficients exacts.
- [ ] même code pour les deux têtes.

#### Indices

1. L'invariance seule accepte l'effondrement.
2. Variance et covariance s'appliquent aux deux vues.
3. Moyenne les termes symétriques avant pondération.

---

### Exercice 6.4 — `CollapseDiagnostics`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/losses.py`. Ce diagnostic est local, pas copié.

#### Contexte

La loss totale ne dit pas directement combien de dimensions se sont effondrées. Le rapport doit montrer l'écart-type moyen, le minimum et la fraction sous un seuil.

#### Objectif

Créer une dataclass et une fonction sans gradient calculant les statistiques par dimension.

#### Entrées

`representations (B,D)` et `threshold` positif.

#### Sorties

Valeurs float `mean_std`, `min_std`, `collapsed_fraction`.

#### Comportement attendu

- calculer std sur la dimension batch.
- convertir seulement les résultats finaux en float.
- compter la proportion std<threshold.
- retourner la dataclass.

#### Exemple

Si 512 dimensions sur 2048 ont std<0,1, `collapsed_fraction=0.25`.

#### Contraintes

- fonction de diagnostic non utilisée dans la loss.
- aucun gradient requis.
- valeurs entre bornes.
- seuil explicite.

#### Code de départ

```python
@dataclass(frozen=True)
class CollapseDiagnostics:
    mean_std: float
    min_std: float
    collapsed_fraction: float


def collapse_diagnostics(representations: Tensor, threshold: float = 0.1) -> CollapseDiagnostics:
    ...
```

#### Étapes conseillées

1. valide B et threshold.
2. entre dans torch.no_grad.
3. calcule std par colonne.
4. agrège et retourne.

#### Vérification manuelle

`PYTHONPATH=src python -c "import torch; from comparison.losses import collapse_diagnostics; print(collapse_diagnostics(torch.zeros(8,4)))"` doit montrer fraction 1.0.

#### Définition de terminé

- [ ] fraction dans [0,1].
- [ ] constante détectée.
- [ ] dataclass en floats.

#### Indices

1. La réduction porte sur dim=0.
2. La comparaison renvoie un masque booléen.
3. La moyenne du masque converti en float donne la fraction.

---

# Phase 7 — Copier LARS et le scheduler

Écris `src/comparison/optim.py` depuis les implémentations de l'exemple Image EB-JEPA.

### Exercice 7.1 — `LARS`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/optim.py`. Source : optimiseur LARS dans `external/eb_jepa/examples/image_jepa/main.py`.

#### Contexte

LARS adapte localement le pas selon les normes du paramètre et de son gradient, puis applique momentum et weight decay. Changer cette logique invaliderait la comparaison officielle.

#### Objectif

Copier intégralement l'optimiseur officiel.

#### Entrées

Groupes de paramètres et hyperparamètres lr, weight_decay, momentum, eta.

#### Sorties

Mise à jour en place des paramètres et état momentum sérialisable.

#### Comportement attendu

- ignorer les gradients absents.
- appliquer les filtres d'exclusion officiels pour biais/normes.
- calculer trust ratio.
- mettre à jour le buffer momentum.
- modifier le paramètre.

#### Exemple

Pour les paramètres matriciels, LARS applique decay et adaptation ; les biais/paramètres 1D suivent les exclusions officielles.

#### Contraintes

- équations exactes.
- état dans optimizer.state.
- pas de branche par architecture.
- compatible state_dict.

#### Code de départ

```python
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.0,
                 momentum: float = 0.9, eta: float = 0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None) -> None: ...
    @torch.no_grad()
    def step(self) -> None: ...
```

#### Étapes conseillées

1. copie les helpers de filtre requis.
2. copie init.
3. copie step ligne par ligne.
4. adapte uniquement imports.

#### Vérification manuelle

`PYTHONPATH=src python - <<'PY'
import torch
from comparison.optim import LARS
p=torch.nn.Parameter(torch.tensor([1.,2.])); p.grad=torch.ones_like(p)
o=LARS([p],lr=.1); o.step(); print(p)
PY` doit modifier p sans NaN.

#### Définition de terminé

- [ ] paramètre modifié.
- [ ] état sérialisable.
- [ ] filtres conservés.
- [ ] aucun NaN.

#### Indices

1. C'est une copie, pas une réinterprétation.
2. Les buffers appartiennent à chaque paramètre.
3. `@torch.no_grad` évite de construire un graphe d'optimisation.

---

### Exercice 7.2 — `WarmupCosineScheduler`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/optim.py`. Source : scheduler de l'exemple officiel ; ajoute seulement sérialisation si absente.

#### Contexte

Le warmup augmente progressivement le LR au début pour éviter une mise à jour instable avec un réseau non calibré. Ensuite, une courbe cosinus le diminue jusqu'au minimum.

#### Objectif

Copier le scheduler, puis rendre sa position sauvegardable/rechargeable.

#### Entrées

Optimizer ; warmup_steps, total_steps, start_lr, base_lr, final_lr ; appel step.

#### Sorties

Nouveau LR écrit dans tous les groupes ; state_dict contenant la progression.

#### Comportement attendu

- pour step<warmup, interpolation linéaire start→base.
- après warmup, interpolation cosinus base→final.
- écrire le même LR dans les groupes.
- incrémenter le compteur.
- sauver/recharger compteur et paramètres.

#### Exemple

Avec 10 warmup steps, le LR monte pendant les dix premiers appels ; il décroît ensuite et atteint le minimum au dernier step.

#### Contraintes

- steps calculés depuis epochs×len(loader).
- aucune remise à zéro par epoch.
- état restauré exactement.
- même scheduler pour les deux têtes.

#### Code de départ

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int,
                 start_lr: float, base_lr: float, final_lr: float) -> None: ...
    def step(self) -> float: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...
```

#### Étapes conseillées

1. dessine les deux segments en nombre de steps.
2. copie la formule officielle.
3. stocke le compteur.
4. ajoute sauvegarde et chargement.

#### Vérification manuelle

`PYTHONPATH=src python - <<'PY'
import torch
from comparison.optim import WarmupCosineScheduler
p=torch.nn.Parameter(torch.tensor(1.)); o=torch.optim.SGD([p],lr=0)
s=WarmupCosineScheduler(o,2,6,0.,1.,0.); print([round(s.step(),3) for _ in range(6)])
PY` doit montrer montée puis descente.

#### Définition de terminé

- [ ] LR continu aux frontières.
- [ ] dernier LR proche final_lr.
- [ ] state_dict reprend au même step.

#### Indices

1. Le warmup est linéaire en compteur global.
2. Le cosinus reçoit une progression entre 0 et 1.
3. Sauvegarde le prochain index de step, pas seulement le LR.

---

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

---

# Phase 9 — Utilitaires et checkpoints

Écris `src/comparison/checkpoint.py`. Un checkpoint doit permettre une reprise identique, pas seulement recharger les poids.

### Exercice 9.1 — `setup_device et setup_seed`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/checkpoint.py`. Sources : helpers de l'exemple officiel.

#### Contexte

La reproductibilité demande de configurer Python, NumPy, PyTorch et les workers. Le device doit être choisi explicitement et BF16 n'est activé que sur CUDA compatible.

#### Objectif

Copier les deux helpers officiels et ajouter une fonction de seed worker déterministe.

#### Entrées

Device demandé et seed entière ; worker_id pour le DataLoader.

#### Sorties

`torch.device` configuré ; générateurs globaux seedés.

#### Comportement attendu

- valider disponibilité CUDA si demandée.
- seeder random, numpy, torch et cuda.
- configurer options déterministes prévues.
- dériver la seed worker depuis torch.initial_seed.

#### Exemple

Deux lancements avec seed 1000 construisent les mêmes poids et le même split ; worker 0 et worker 1 reçoivent des seeds différentes.

#### Contraintes

- aucun fallback silencieux A100→CPU pour benchmark.
- seed dans plage NumPy.
- appel avant construction modèle/données.

#### Code de départ

```python
def setup_device(device_name: str) -> torch.device: ...
def setup_seed(seed: int) -> None: ...
def seed_worker(worker_id: int) -> None: ...
```

#### Étapes conseillées

1. copie les deux helpers.
2. ajoute Python/NumPy si nécessaire.
3. écris seed_worker.
4. appelle-les dans cet ordre plus tard.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.checkpoint import setup_seed; import torch; setup_seed(7); a=torch.rand(1); setup_seed(7); print(torch.equal(a,torch.rand(1)))"` affiche `True`.

#### Définition de terminé

- [ ] tirage reproductible.
- [ ] workers distincts.
- [ ] device indisponible signalé.

#### Indices

1. Une seed doit précéder les constructeurs aléatoires.
2. torch.initial_seed inclut l'identité worker.
3. Le benchmark a des exigences plus strictes que les contrôles CPU.

---

### Exercice 9.2 — `save_checkpoint`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/checkpoint.py`. Source : sauvegarde officielle adaptée aux objets de ce projet.

#### Contexte

Sauver seulement le modèle empêcherait de reprendre momentum, LR et progression. Le meilleur checkpoint doit aussi identifier architecture et configuration.

#### Objectif

Sérialiser tous les états nécessaires dans un fichier écrit de façon sûre.

#### Entrées

Path, model, optimizer, scheduler, epoch, best metric, cfg et head_type.

#### Sorties

Fichier `.pt` contenant états et métadonnées.

#### Comportement attendu

- créer le dossier parent.
- construire un dict avec model/optimizer/scheduler.
- ajouter epoch, best, config résolue, head_type, seed.
- écrire vers un fichier temporaire puis remplacer la cible.

#### Exemple

Après epoch 50, le checkpoint contient le prochain epoch à exécuter, les buffers LARS et le compteur exact du scheduler.

#### Contraintes

- scheduler obligatoire.
- configuration sérialisable.
- pas de poids seuls.
- remplacement atomique dans le même dossier.

#### Code de départ

```python
def save_checkpoint(path: str | Path, *, model: ImageSSL, optimizer: Optimizer,
                    scheduler: WarmupCosineScheduler, epoch: int,
                    best_validation: float, cfg: DictConfig) -> None:
    ...
```

#### Étapes conseillées

1. résous cfg en conteneur Python.
2. compose le payload.
3. sauve dans path.tmp.
4. remplace la cible.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.checkpoint import save_checkpoint; print(save_checkpoint.__annotations__)"` doit montrer scheduler et cfg dans la signature.

#### Définition de terminé

- [ ] toutes les clés présentes.
- [ ] parent créé.
- [ ] écriture atomique.
- [ ] checkpoint chargeable par torch.load.

#### Indices

1. OmegaConf.to_container résout le DictConfig.
2. `Path.replace` effectue le remplacement.
3. Sauve les state_dict, pas les objets complets.

---

### Exercice 9.3 — `load_checkpoint`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/checkpoint.py`, symétrique à save_checkpoint.

#### Contexte

Une reprise avec la mauvaise tête ou des dimensions différentes peut parfois charger partiellement et produire une expérience invalide. Le chargement doit vérifier l'identité du run avant mutation.

#### Objectif

Valider les métadonnées, charger strictement tous les états et retourner la progression.

#### Entrées

Path, objets déjà construits, cfg courante, device.

#### Sorties

Dataclass ou tuple contenant start_epoch et best_validation.

#### Comportement attendu

- charger avec map_location.
- comparer head_type et dimensions critiques avant load_state_dict.
- charger modèle strict=True.
- charger optimizer et scheduler.
- retourner epoch suivant et meilleur score.

#### Exemple

Un checkpoint baseline refusé par un modèle cortical avant que ses poids ne soient modifiés ; un checkpoint compatible reprend au step LR exact.

#### Contraintes

- aucun strict=False.
- aucun état manquant ignoré.
- message explicite sur mismatch.
- restauration scheduler obligatoire.

#### Code de départ

```python
@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_validation: float


def load_checkpoint(path: str | Path, *, model: ImageSSL, optimizer: Optimizer,
                    scheduler: WarmupCosineScheduler, cfg: DictConfig,
                    device: torch.device) -> ResumeState:
    ...
```

#### Étapes conseillées

1. charge d'abord le payload sans muter.
2. valide métadonnées/config.
3. charge les trois state_dict.
4. construis ResumeState.

#### Vérification manuelle

Sauve un checkpoint miniature, modifie les poids, recharge-le et vérifie manuellement que poids, compteur scheduler et epoch retrouvent leurs valeurs.

#### Définition de terminé

- [ ] mismatch refusé avant mutation.
- [ ] chargement strict.
- [ ] optimizer/scheduler restaurés.
- [ ] epoch suivant correct.

#### Indices

1. Les métadonnées se lisent avant les poids.
2. `map_location=device` évite un chargement sur le mauvais GPU.
3. Si epoch sauvegardé est le dernier fini, start_epoch vaut epoch+1.

---

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

---

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

---

# Phase 12 — Mesurer le compute sur A100

Écris `src/comparison/benchmark.py`. Les entrées synthétiques empêchent le DataLoader de polluer les mesures.

### Exercice 12.1 — `ComputeMetrics`

**Difficulté :** Facile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

Le rapport exige paramètres, FLOPs, latence, débit et mémoire avec des unités non ambiguës.

#### Objectif

Créer une dataclass immuable pour un périmètre de mesure.

#### Entrées

Cinq nombres calculés par les helpers suivants.

#### Sorties

Un objet sérialisable.

#### Comportement attendu

- déclarer parameters.
- déclarer flops_per_sample.
- déclarer latency_ms et throughput_samples_s.
- déclarer peak_memory_bytes.

#### Exemple

Une latence de 2 ms pour un batch de 256 correspond à 128 000 samples/s.

#### Contraintes

- valeurs non négatives.
- unités visibles dans les noms.
- aucun Tensor.

#### Code de départ

```python
@dataclass(frozen=True)
class ComputeMetrics:
    parameters: int
    flops_per_sample: float
    latency_ms: float
    throughput_samples_s: float
    peak_memory_bytes: int
```

#### Étapes conseillées

1. importe dataclass.
2. déclare la classe.
3. réutilise-la pour tête et modèle complet.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.benchmark import ComputeMetrics; print(ComputeMetrics.__annotations__)"` doit réussir.

#### Définition de terminé

- [ ] cinq métriques présentes.
- [ ] unités explicites.
- [ ] types simples.

#### Indices

1. Les octets seront convertis dans le rapport.
2. Les FLOPs sont par sample.
3. Le débit dépend du batch fixé.

---

### Exercice 12.2 — `measure_flops`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/benchmark.py`, avec `fvcore.nn.FlopCountAnalysis`.

#### Contexte

Le compteur doit mesurer exactement le module reçu, avec un Tensor déjà placé sur le device.

#### Objectif

Compter les opérations puis normaliser par la taille du batch.

#### Entrées

`module: nn.Module`, `example_input: Tensor (B,...)`.

#### Sorties

Un float en FLOPs par sample.

#### Comportement attendu

- passer le module en eval.
- construire FlopCountAnalysis avec un tuple d'entrées.
- lire total.
- diviser par B.
- laisser visibles les opérateurs non supportés.

#### Exemple

Un total de 2e9 pour B=256 devient 7.8125e6 FLOPs/sample.

#### Contraintes

- même forme d'entrée pour les deux têtes.
- aucun DataLoader.
- aucun avertissement masqué.

#### Code de départ

```python
def measure_flops(module: nn.Module, example_input: Tensor) -> float:
    ...
```

#### Étapes conseillées

1. construis l'analyse.
2. lis total.
3. normalise.

#### Vérification manuelle

Mesure `nn.Linear(512,2048)` avec B=4 puis B=8 ; les valeurs par sample doivent coïncider.

#### Définition de terminé

- [ ] résultat positif.
- [ ] normalisation par sample.
- [ ] module inchangé.

#### Indices

1. fvcore attend les entrées dans un tuple.
2. Le batch multiplie le total.
3. Les modules custom peuvent produire des avertissements utiles.

---

### Exercice 12.3 — `measure_latency`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

CUDA est asynchrone. Les Events et synchronisations mesurent les kernels ; 100 warmups stabilisent caches et fréquences avant 500 mesures.

#### Objectif

Mesurer latence et débit sur A100.

#### Entrées

Callable sans argument, batch_size, warmups=100, measurements=500.

#### Sorties

Tuple `(latency_ms, throughput_samples_s)`.

#### Comportement attendu

- exécuter 100 warmups.
- synchroniser.
- enregistrer start/end avec CUDA Events sur 500 appels.
- synchroniser avant lecture.
- agréger et convertir en débit.

#### Exemple

2 ms pour B=256 donne 128 000 samples/s.

#### Contraintes

- exactement 100 et 500.
- CUDA Events enable_timing.
- pas de time.time.
- même BF16.

#### Code de départ

```python
def measure_latency(call: Callable[[], Tensor], batch_size: int,
                    warmups: int = 100, measurements: int = 500) -> tuple[float, float]:
    ...
```

#### Étapes conseillées

1. effectue warmup.
2. enregistre les Events.
3. synchronise.
4. calcule latence et débit.

#### Vérification manuelle

Sur A100, vérifie que les 500 durées sont positives et que le débit est fini.

#### Définition de terminé

- [ ] warmups exacts.
- [ ] mesures exactes.
- [ ] synchronisation.
- [ ] unités correctes.

#### Indices

1. elapsed_time retourne des millisecondes.
2. Synchronise avant la lecture.
3. Débit = B/(ms/1000).

---

### Exercice 12.4 — `measure_peak_memory`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

PyTorch conserve des allocations. Il faut réinitialiser la statistique de pic juste avant le callable.

#### Objectif

Mesurer le pic CUDA sous no_grad.

#### Entrées

Un callable représentant la tête ou le modèle complet.

#### Sorties

Un entier exprimé en bytes.

#### Comportement attendu

- synchroniser.
- reset_peak_memory_stats.
- exécuter sous no_grad.
- synchroniser.
- lire max_memory_allocated.

#### Exemple

Le même batch et le même dtype sont utilisés pour baseline et cortical.

#### Contraintes

- unité bytes.
- aucun DataLoader.
- protocole strictement identique.

#### Code de départ

```python
def measure_peak_memory(call: Callable[[], Tensor]) -> int:
    ...
```

#### Étapes conseillées

1. réinitialise.
2. exécute.
3. synchronise.
4. lis le compteur.

#### Vérification manuelle

Sur A100, le résultat doit être positif et stable à quelques allocations près.

#### Définition de terminé

- [ ] reset juste avant.
- [ ] no_grad.
- [ ] synchronisation.
- [ ] entier bytes.

#### Indices

1. Les statistiques sont par device.
2. allocated diffère de reserved.
3. Documente pic absolu ou delta.

---

### Exercice 12.5 — `benchmark_model`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

Cette fonction refuse un autre GPU afin que la revendication A100 soit exacte. Elle sépare tête seule et modèle complet.

#### Objectif

Assembler toutes les mesures en BF16.

#### Entrées

`model: ImageSSL`, cfg validée et device CUDA.

#### Sorties

BenchmarkResult contenant deux ComputeMetrics et les métadonnées.

#### Comportement attendu

- vérifier A100 dans le nom du GPU.
- passer en eval et BF16.
- créer 256 images synthétiques.
- calculer les features hors chronométrage de tête.
- mesurer tête puis modèle.
- retourner GPU, dtype et batch.

#### Exemple

Les deux architectures reçoivent les mêmes formes ; seule la tête diffère.

#### Contraintes

- A100 obligatoire.
- batch 256 sans réduction après OOM.
- 100/500.
- BF16.
- aucun dataset.

#### Code de départ

```python
def benchmark_model(model: ImageSSL, cfg: DictConfig,
                    device: torch.device) -> BenchmarkResult:
    ...
```

#### Étapes conseillées

1. valide GPU.
2. prépare images et features.
3. définis les deux callables.
4. appelle les helpers.
5. compose le résultat.

#### Vérification manuelle

Sur A100, vérifie le nom du GPU, B=256, BF16 et des métriques positives.

#### Définition de terminé

- [ ] refus non-A100.
- [ ] deux périmètres.
- [ ] mêmes entrées.
- [ ] BF16.

#### Indices

1. `torch.cuda.get_device_name` donne le modèle.
2. Les features de tête sont calculées hors chrono.
3. Ne charge jamais CIFAR ici.

---

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

---

# Phase 14 — Écrire les quatre entrées CLI

Crée quatre scripts minces. Toute logique scientifique reste sous `src/comparison`.

### Exercice 14.1 — `Scripts train, evaluate, benchmark et report`

**Difficulté :** Moyen

#### Où travailler

`scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py` et `scripts/report.py`.

#### Contexte

Les scripts offrent une interface reproductible. Toute logique reste dans `src/comparison` afin qu'un import Python et un appel CLI utilisent le même protocole.

#### Objectif

Écrire quatre fonctions `main()` avec argparse et un bloc d'entrée standard.

#### Entrées

Train : config, seed, output, resume, overrides. Evaluate : config, checkpoint, cinq pair seeds, output. Benchmark : config, checkpoint, output. Report : six JSON et deux outputs.

#### Sorties

Code de sortie 0 et chemin du fichier produit affiché.

#### Comportement attendu

- analyser seulement les arguments.
- charger la configuration avec load_config.
- appeler exactement une fonction métier.
- afficher le résultat.
- protéger l'appel par le bloc main.

#### Exemple

`python scripts/train.py --config configs/baseline.yaml --seed 1 --output-dir runs/baseline/1 optimization.epochs=1` appelle `comparison.train.run`.

#### Contraintes

- quatre scripts.
- aucune boucle d'entraînement dupliquée.
- aucun fallback après OOM.
- architecture lue dans la config.

#### Code de départ

```python
# scripts/train.py
import argparse
from comparison.config import load_config
from comparison.train import run


def main() -> None:
    ...


if __name__ == "__main__":
    main()

# Reprends cette structure pour evaluate.py, benchmark.py et report.py.
```

#### Étapes conseillées

1. définis les quatre parsers.
2. charge et valide les arguments.
3. appelle les fonctions métier.
4. affiche les chemins.

#### Vérification manuelle

Les quatre commandes `python scripts/<nom>.py --help` doivent fonctionner sans télécharger CIFAR ni initialiser CUDA.

#### Définition de terminé

- [ ] quatre aides fonctionnent.
- [ ] scripts minces.
- [ ] invocations documentées.
- [ ] aucune logique scientifique dupliquée.

#### Indices

1. `nargs='*'` accepte les overrides pointés.
2. Importer un script n'exécute pas main.
3. Toutes les validations importantes existent déjà sous src.
