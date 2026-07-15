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

