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
- aucun symbole EB-JEPA n’est encore copié dans `src/` ;
- le tableau ci-dessous constitue la liste de travail pour les phases suivantes.

## Symboles EB-JEPA retenus

| Symbole local prévu | Source officielle | Traitement prévu |
|---|---|---|
| augmentations CIFAR-10 | `examples/image_jepa/dataset.py:10-96` | copier avec attribution |
| `PairedViewDataset` | `examples/image_jepa/dataset.py:99-113` | adapter pour supprimer le label |
| `ResNet18` | `examples/image_jepa/main.py:61-75` | copier sans changement architectural |
| `MLPProjector` | `examples/image_jepa/main.py:78-102` | extraire le bloc `projector` |
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
