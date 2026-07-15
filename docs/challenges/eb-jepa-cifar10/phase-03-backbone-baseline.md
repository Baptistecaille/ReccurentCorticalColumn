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

