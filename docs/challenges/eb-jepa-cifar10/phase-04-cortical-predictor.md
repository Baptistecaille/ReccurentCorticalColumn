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

