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

