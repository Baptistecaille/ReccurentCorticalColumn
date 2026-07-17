# Prise en charge de PyTorch MPS

## Objectif

Permettre l'entraînement et l'évaluation du comparatif EB-JEPA CIFAR-10 sur
les Mac Apple Silicon via le backend PyTorch MPS, sans modifier le protocole de
benchmark qui reste strictement réservé à une NVIDIA A100 avec CUDA.

## Sélection du périphérique

`setup_device("auto")` sélectionne le premier backend disponible dans cet
ordre : CUDA, MPS, puis CPU. La fonction accepte aussi les demandes explicites
`"cuda"`, `"cuda:<index>"`, `"mps"` et `"cpu"`.

Une demande explicite de MPS échoue avec un message clair lorsque PyTorch n'a
pas accès au backend MPS. Les erreurs CUDA existantes restent inchangées. Le
benchmark continue d'exiger un périphérique CUDA et rejette donc MPS comme il
rejette déjà CPU.

## Capacités dépendantes du périphérique

La logique liée aux capacités matérielles est centralisée dans de petits
helpers testables :

- la précision mixte `bfloat16` demandée par la configuration est désactivée
  sur MPS ; CUDA et CPU conservent leur comportement actuel ;
- `pin_memory` est désactivé sur MPS même si la configuration le demande ;
  CUDA et CPU continuent de respecter la valeur configurée.

Le code d'entraînement et d'évaluation utilise ces helpers au lieu de déduire
les capacités uniquement de la configuration. Les DataLoaders reçoivent le
périphérique sélectionné afin de construire leurs options sans avertissement
MPS. Le transfert des tenseurs, le modèle, les checkpoints et les calculs
restent pilotés par le même objet `torch.device`.

## Reproductibilité et checkpoints

Les états RNG Python, NumPy, PyTorch CPU et CUDA conservent leur format actuel.
PyTorch MPS est initialisé par `torch.manual_seed`, déjà appelé par
`setup_seed`; aucun nouveau champ de checkpoint n'est ajouté. Les checkpoints
restent chargeables entre CPU, CUDA et MPS grâce au `map_location` existant.

## Tests

Les tests unitaires couvrent :

- la priorité CUDA → MPS → CPU en mode automatique ;
- la sélection explicite de MPS lorsqu'il est disponible ;
- l'erreur explicite lorsqu'il ne l'est pas ;
- la désactivation de `bfloat16` et de `pin_memory` sur MPS ;
- la conservation des comportements CUDA et CPU existants ;
- l'utilisation effective de ces décisions par les DataLoaders et les chemins
  d'entraînement ou d'évaluation concernés.

Les tests simulent les capacités matérielles pour rester exécutables sur une
machine sans CUDA ou sans MPS. La suite existante complète est ensuite exécutée
pour détecter les régressions.

## Documentation

Le README du comparatif décrit la sélection automatique, la possibilité de
faire tourner Train et Evaluate sur MPS, ainsi que le maintien du benchmark
A100/CUDA. Il précise que MPS utilise la précision standard et désactive la
mémoire épinglée.

## Hors périmètre

- rendre le benchmark portable sur MPS ;
- changer le protocole scientifique A100/bfloat16 du benchmark ;
- ajouter un champ `device` aux fichiers YAML ou aux cellules du notebook ;
- garantir que chaque opérateur futur de PyTorch possède une implémentation
  MPS.
