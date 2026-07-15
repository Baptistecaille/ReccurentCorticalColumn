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

