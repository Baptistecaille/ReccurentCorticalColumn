# Rapport de correction — Phases 0 à 4

Date : 15 juillet 2026

## Périmètre

Ce rapport couvre uniquement les composants des phases 0 à 4. Aucun composant
de la phase 5 (`ImageSSL`, `build_model`, `ParameterCounts`) n'a été implémenté.
Aucun test automatisé, fichier de test ou appel à `pytest` n'a été ajouté ou
exécuté.

## Phase 0 — Projet autonome

### `src/comparison/__init__.py`

- `ImageSSL` est désormais associé au futur module `.model` au lieu de `.heads`.
- `build_model` est désormais associé au futur module `.model`.
- `build_head` reste associé à `.heads`.
- Le chargement reste paresseux : importer `comparison` ne nécessite pas que les
  modules des phases suivantes existent déjà.

### `pyproject.toml`

- La version minimale de setuptools est alignée sur la roadmap : `>=68.0`.
- Les six dépendances runtime existantes sont conservées.
- Aucune dépendance de test n'a été ajoutée.

### `SOURCES.md`

- La mention obsolète « aucun symbole EB-JEPA n'est copié » a été retirée.
- Les augmentations, `PairedViewDataset`, `ResNet18` et `MLPProjector` sont
  enregistrés avec leur fichier local, leur source et leurs adaptations.
- Les symboles des phases suivantes restent dans une table séparée « planifiés ».

## Phase 1 — Augmentations

### `src/comparison/augmentation.py`

- Un en-tête de provenance indique le dépôt, le fichier et le commit source.
- Les transformations et leur ordre officiel sont conservés.
- `get_train_transforms` accepte maintenant `crop_scale` afin que la
  configuration puisse injecter cet intervalle sans modifier le pipeline.
- La sortie reste un tenseur normalisé de forme `(3, 32, 32)`.

### Roadmap

- Le paramètre `ratio` a été retiré du squelette de `RandomResizedCrop`, car la
  classe officielle copiée n'expose que `size` et `scale`.
- Le blur a été retiré de l'exercice `get_train_transforms` : il ne figure pas
  dans `examples/image_jepa/dataset.py:70-86` au commit de référence.
- L'ordre documenté correspond maintenant à la source : crop, jitter,
  grayscale, solarisation, flip, conversion Tensor, normalisation.

## Phase 2 — Données et splits

### `src/comparison/data.py`

Le fichier a été remis en conformité avec les interfaces de la roadmap :

- suppression de l'import absolu invalide `from augmentation ...` ;
- injection des transformations dans `make_dataloaders` ;
- `PairedViewDataset` accepte un dataset retournant une image seule ou
  `(image, label)` et ne retourne jamais le label ;
- suppression de l'argument inutile `num_crops` : cette classe produit toujours
  exactement deux vues ;
- renommage de `build_split_indicies` en `build_split_indices` ;
- nouvelle signature `(dataset_size, validation_size, split_seed)` ;
- validation explicite des tailles du split ;
- permutation obtenue avec un `torch.Generator` local, sans modifier la seed
  globale ;
- `DeterministicPairDataset` utilise maintenant l'interface `pair_seed` ;
- les états aléatoires Python, NumPy et PyTorch CPU sont restaurés dans un bloc
  `finally`, y compris si une transformation échoue ;
- `DataLoaders` redevient un simple conteneur immuable ;
- `make_dataloaders(cfg, train_transform, eval_transform)` construit les trois
  datasets et contrôle les tailles 45 000/5 000/10 000 ;
- `shuffle=True` et `drop_last=True` sont réservés au train ;
- validation et test utilisent des paires déterministes et ne sont pas mélangés.

Les états CUDA ne sont volontairement pas modifiés par le dataset : les
augmentations PIL/torchvision de cette phase s'exécutent sur CPU dans les workers
du DataLoader.

## Phase 3 — Backbone et baseline

### `src/comparison/backbone.py`

- `ResNet18` reste conforme à la classe officielle : convolution initiale 3×3,
  stride 1, padding 2, aucun max-pool et aucune couche de classification.
- `weights=None` est maintenant explicite.
- `features_dim = 512` est déclaré comme attribut de classe.
- `ImageSSL` et le projecteur inline ont été retirés : ils mélangeaient les
  responsabilités des phases 3 et 5.

### `src/comparison/heads.py`

- L'import du module inexistant `comparison.mlp` a été supprimé.
- `MLPProjector` est défini directement dans ce fichier.
- Son architecture est celle du projecteur officiel : trois `Linear`, deux
  `BatchNorm1d`, deux `ReLU`, sans activation finale.
- Un en-tête de provenance documente l'extraction depuis EB-JEPA.

## Phase 4 — Tête corticale

### Cœur cortical 4.1 à 4.6

- Aucun changement : les six modules copiés étaient déjà conformes.
- Le chargement strict des poids de la source locale dans la copie produisait
  déjà des sorties exactement identiques.

### `CorticalHead`

- Les `assert` ont été remplacés par des `ValueError` explicites.
- L'entrée doit avoir la forme `(batch, input_dim)`.
- La sortie du prédicteur doit avoir la forme `(batch, output_dim)`.
- Aucune couche entraînable supplémentaire n'a été ajoutée.

### `build_head`

- La factory lit désormais `cfg.model.head_type`.
- La branche baseline utilise `model.feature_dim` et `model.output_dim`.
- La dimension cachée du MLP utilise `model.mlp_hidden_dim` si elle existe,
  sinon `model.output_dim`, soit 2048 avec la configuration imposée.
- La branche corticale construit `FixedTreePredictor`, puis `CorticalHead`.
- Un type inconnu produit une `ValueError` mentionnant `model.head_type`.

## Contrôles manuels effectués

Ces contrôles ne constituent pas une suite de tests :

- compilation syntaxique de `src/comparison` : réussie ;
- import de `comparison`, `comparison.data` et `comparison.heads` : réussi ;
- augmentation PIL 32×32 : sortie `(3, 32, 32)`, valeurs finies ;
- split 50 000 : 45 000 train, 5 000 validation, 50 000 indices uniques ;
- paires déterministes : lectures répétées identiques ;
- états RNG Python, NumPy et PyTorch restaurés après lecture ;
- backbone : `(2, 3, 32, 32) → (2, 512)` ;
- projecteur MLP : `(2, 512) → (2, 2048)` ;
- factory baseline : retourne `MLPProjector` ;
- factory corticale : retourne `CorticalHead` et produit `(2, 2048)` ;
- mauvaise entrée corticale `(2, 511)` : `ValueError` explicite.

## État avant la phase 5

Les phases 0 à 4 respectent maintenant la séparation suivante :

```text
augmentation.py  -> transformations
data.py          -> datasets, splits et loaders
backbone.py      -> ResNet18 uniquement
heads.py         -> MLPProjector, CorticalHead et build_head
cortical/        -> FixedTreePredictor et ses modules
model.py         -> à créer pendant la phase 5
```
