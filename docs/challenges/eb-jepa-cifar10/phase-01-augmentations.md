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

