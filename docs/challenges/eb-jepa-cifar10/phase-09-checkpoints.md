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

