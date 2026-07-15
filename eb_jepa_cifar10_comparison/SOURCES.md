# Registre des sources copiées

## Référence

- Dépôt : `https://github.com/facebookresearch/eb_jepa.git`
- Branche : `main`
- Commit : `966e61e9285b3a876f49b9774e9720d9a99a7925`
- Licence : Apache License 2.0 (`external/eb_jepa/LICENSE.md`)
- Clone local intact : `external/eb_jepa/`
- Copie locale de la licence : `eb_jepa_cifar10_comparison/LICENSE.md`

## État de la provenance

- Phase 0 : licence officielle copiée à l’identique ;
- phases 1 à 4 : augmentations, datasets, backbone, projecteur MLP et prédicteur
  cortical présents dans `src/comparison/` ;
- les phases suivantes restent des éléments planifiés.

## Symboles EB-JEPA copiés

| Symbole local | Fichier local | Source officielle | Adaptations |
|---|---|---|---|
| augmentations CIFAR-10 | `src/comparison/augmentation.py` | `examples/image_jepa/dataset.py:10-86` | attribution et paramètre `crop_scale` injectable |
| `PairedViewDataset` | `src/comparison/data.py` | `examples/image_jepa/dataset.py:99-113` | exactement deux vues, label supprimé |
| `ResNet18` | `src/comparison/backbone.py` | `examples/image_jepa/main.py:61-75` | annotations et `weights=None` explicite |
| `MLPProjector` | `src/comparison/heads.py` | `examples/image_jepa/main.py:78-102` | bloc `projector` extrait dans une classe nommée |

## Symboles EB-JEPA planifiés

| Symbole local prévu | Source officielle | Traitement prévu |
|---|---|---|
| `LARS` | `examples/image_jepa/main.py:105-207` | copier avec attribution |
| `WarmupCosineScheduler` | `examples/image_jepa/main.py:210-247` | copier avec attribution |
| `HingeStdLoss` | `eb_jepa/losses.py:56-81` | copier avec attribution |
| `CovarianceLoss` | `eb_jepa/losses.py:84-111` | copier avec attribution |
| `VICRegLoss` | `eb_jepa/losses.py:306-342` | copier avec attribution |
| `setup_device` | `eb_jepa/training_utils.py:18-25` | copier avec attribution |
| `setup_seed` | `eb_jepa/training_utils.py:27-35` | copier avec attribution |
| `save_checkpoint` | `eb_jepa/training_utils.py:146-176` | copier puis ajouter la configuration |
| `load_checkpoint` | `eb_jepa/training_utils.py:179-238` | copier puis vérifier le type de tête |
| `load_config` | `eb_jepa/training_utils.py:241-269` | copier avec attribution |
| `train_epoch` | `examples/image_jepa/main.py:250-332` | copier puis retirer linear probe et labels |

## Code explicitement non repris

- `examples/image_jepa/eval.py` : linear probe exclu du protocole ;
- `BCS`, SIGReg, pertes temporelles et inverse dynamics : hors périmètre ;
- launchers SLURM et intégration W&B : non nécessaires à la comparaison locale ;
- code ViT : le protocole commun retenu utilise ResNet-18.

## Code local

Les fichiers de `Predictor/` sont du code local, pas du code EB-JEPA. Leur copie
dans le nouveau projet doit préserver leurs modules séparés et leurs imports,
sans copier le dossier `Predictor/tests/`.

| Symbole local copié | Source locale | Adaptations |
|---|---|---|
| arbre et `ColumnNode` | `Predictor/utils/tree_structure.py` | retrait de `pdb`, annotation de retour |
| `FiLMModulation` | `Predictor/modules/modulation.py` | annotations uniquement |
| `_lateral_attention`, `ColumnStep`, `CorticalColumn` | `Predictor/modules/column.py` | correction de `torch.torch.Tensor`, noms locaux clarifiés |
| `ChildDecomposition`, `FeedbackProjection` | `Predictor/modules/decomposition.py` | annotations et formatage uniquement |
| projections et intégration récursive | `Predictor/modules/integration.py` | annotations et formatage uniquement |
| `FixedTreePredictor` | `Predictor/model/fixed_tree_predictor.py` | imports relatifs, retrait du bloc exécutable |
