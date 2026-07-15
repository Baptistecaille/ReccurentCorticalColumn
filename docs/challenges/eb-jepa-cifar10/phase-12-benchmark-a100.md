# Phase 12 — Mesurer le compute sur A100

Écris `src/comparison/benchmark.py`. Les entrées synthétiques empêchent le DataLoader de polluer les mesures.

### Exercice 12.1 — `ComputeMetrics`

**Difficulté :** Facile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

Le rapport exige paramètres, FLOPs, latence, débit et mémoire avec des unités non ambiguës.

#### Objectif

Créer une dataclass immuable pour un périmètre de mesure.

#### Entrées

Cinq nombres calculés par les helpers suivants.

#### Sorties

Un objet sérialisable.

#### Comportement attendu

- déclarer parameters.
- déclarer flops_per_sample.
- déclarer latency_ms et throughput_samples_s.
- déclarer peak_memory_bytes.

#### Exemple

Une latence de 2 ms pour un batch de 256 correspond à 128 000 samples/s.

#### Contraintes

- valeurs non négatives.
- unités visibles dans les noms.
- aucun Tensor.

#### Code de départ

```python
@dataclass(frozen=True)
class ComputeMetrics:
    parameters: int
    flops_per_sample: float
    latency_ms: float
    throughput_samples_s: float
    peak_memory_bytes: int
```

#### Étapes conseillées

1. importe dataclass.
2. déclare la classe.
3. réutilise-la pour tête et modèle complet.

#### Vérification manuelle

`PYTHONPATH=src python -c "from comparison.benchmark import ComputeMetrics; print(ComputeMetrics.__annotations__)"` doit réussir.

#### Définition de terminé

- [ ] cinq métriques présentes.
- [ ] unités explicites.
- [ ] types simples.

#### Indices

1. Les octets seront convertis dans le rapport.
2. Les FLOPs sont par sample.
3. Le débit dépend du batch fixé.

---

### Exercice 12.2 — `measure_flops`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/benchmark.py`, avec `fvcore.nn.FlopCountAnalysis`.

#### Contexte

Le compteur doit mesurer exactement le module reçu, avec un Tensor déjà placé sur le device.

#### Objectif

Compter les opérations puis normaliser par la taille du batch.

#### Entrées

`module: nn.Module`, `example_input: Tensor (B,...)`.

#### Sorties

Un float en FLOPs par sample.

#### Comportement attendu

- passer le module en eval.
- construire FlopCountAnalysis avec un tuple d'entrées.
- lire total.
- diviser par B.
- laisser visibles les opérateurs non supportés.

#### Exemple

Un total de 2e9 pour B=256 devient 7.8125e6 FLOPs/sample.

#### Contraintes

- même forme d'entrée pour les deux têtes.
- aucun DataLoader.
- aucun avertissement masqué.

#### Code de départ

```python
def measure_flops(module: nn.Module, example_input: Tensor) -> float:
    ...
```

#### Étapes conseillées

1. construis l'analyse.
2. lis total.
3. normalise.

#### Vérification manuelle

Mesure `nn.Linear(512,2048)` avec B=4 puis B=8 ; les valeurs par sample doivent coïncider.

#### Définition de terminé

- [ ] résultat positif.
- [ ] normalisation par sample.
- [ ] module inchangé.

#### Indices

1. fvcore attend les entrées dans un tuple.
2. Le batch multiplie le total.
3. Les modules custom peuvent produire des avertissements utiles.

---

### Exercice 12.3 — `measure_latency`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

CUDA est asynchrone. Les Events et synchronisations mesurent les kernels ; 100 warmups stabilisent caches et fréquences avant 500 mesures.

#### Objectif

Mesurer latence et débit sur A100.

#### Entrées

Callable sans argument, batch_size, warmups=100, measurements=500.

#### Sorties

Tuple `(latency_ms, throughput_samples_s)`.

#### Comportement attendu

- exécuter 100 warmups.
- synchroniser.
- enregistrer start/end avec CUDA Events sur 500 appels.
- synchroniser avant lecture.
- agréger et convertir en débit.

#### Exemple

2 ms pour B=256 donne 128 000 samples/s.

#### Contraintes

- exactement 100 et 500.
- CUDA Events enable_timing.
- pas de time.time.
- même BF16.

#### Code de départ

```python
def measure_latency(call: Callable[[], Tensor], batch_size: int,
                    warmups: int = 100, measurements: int = 500) -> tuple[float, float]:
    ...
```

#### Étapes conseillées

1. effectue warmup.
2. enregistre les Events.
3. synchronise.
4. calcule latence et débit.

#### Vérification manuelle

Sur A100, vérifie que les 500 durées sont positives et que le débit est fini.

#### Définition de terminé

- [ ] warmups exacts.
- [ ] mesures exactes.
- [ ] synchronisation.
- [ ] unités correctes.

#### Indices

1. elapsed_time retourne des millisecondes.
2. Synchronise avant la lecture.
3. Débit = B/(ms/1000).

---

### Exercice 12.4 — `measure_peak_memory`

**Difficulté :** Moyen

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

PyTorch conserve des allocations. Il faut réinitialiser la statistique de pic juste avant le callable.

#### Objectif

Mesurer le pic CUDA sous no_grad.

#### Entrées

Un callable représentant la tête ou le modèle complet.

#### Sorties

Un entier exprimé en bytes.

#### Comportement attendu

- synchroniser.
- reset_peak_memory_stats.
- exécuter sous no_grad.
- synchroniser.
- lire max_memory_allocated.

#### Exemple

Le même batch et le même dtype sont utilisés pour baseline et cortical.

#### Contraintes

- unité bytes.
- aucun DataLoader.
- protocole strictement identique.

#### Code de départ

```python
def measure_peak_memory(call: Callable[[], Tensor]) -> int:
    ...
```

#### Étapes conseillées

1. réinitialise.
2. exécute.
3. synchronise.
4. lis le compteur.

#### Vérification manuelle

Sur A100, le résultat doit être positif et stable à quelques allocations près.

#### Définition de terminé

- [ ] reset juste avant.
- [ ] no_grad.
- [ ] synchronisation.
- [ ] entier bytes.

#### Indices

1. Les statistiques sont par device.
2. allocated diffère de reserved.
3. Documente pic absolu ou delta.

---

### Exercice 12.5 — `benchmark_model`

**Difficulté :** Difficile

#### Où travailler

`src/comparison/benchmark.py`.

#### Contexte

Cette fonction refuse un autre GPU afin que la revendication A100 soit exacte. Elle sépare tête seule et modèle complet.

#### Objectif

Assembler toutes les mesures en BF16.

#### Entrées

`model: ImageSSL`, cfg validée et device CUDA.

#### Sorties

BenchmarkResult contenant deux ComputeMetrics et les métadonnées.

#### Comportement attendu

- vérifier A100 dans le nom du GPU.
- passer en eval et BF16.
- créer 256 images synthétiques.
- calculer les features hors chronométrage de tête.
- mesurer tête puis modèle.
- retourner GPU, dtype et batch.

#### Exemple

Les deux architectures reçoivent les mêmes formes ; seule la tête diffère.

#### Contraintes

- A100 obligatoire.
- batch 256 sans réduction après OOM.
- 100/500.
- BF16.
- aucun dataset.

#### Code de départ

```python
def benchmark_model(model: ImageSSL, cfg: DictConfig,
                    device: torch.device) -> BenchmarkResult:
    ...
```

#### Étapes conseillées

1. valide GPU.
2. prépare images et features.
3. définis les deux callables.
4. appelle les helpers.
5. compose le résultat.

#### Vérification manuelle

Sur A100, vérifie le nom du GPU, B=256, BF16 et des métriques positives.

#### Définition de terminé

- [ ] refus non-A100.
- [ ] deux périmètres.
- [ ] mêmes entrées.
- [ ] BF16.

#### Indices

1. `torch.cuda.get_device_name` donne le modèle.
2. Les features de tête sont calculées hors chrono.
3. Ne charge jamais CIFAR ici.

