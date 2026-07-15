# Conception — Roadmap au format LeetCode appliqué à un projet

## Objectif

Réécrire toute la roadmap d'implémentation EB-JEPA/CIFAR-10 afin qu'une personne
qui découvre le dépôt puisse commencer chaque exercice sans deviner le fichier à
ouvrir, le code déjà disponible, les formes des tenseurs ou la place du composant
dans l'architecture.

La nouvelle roadmap couvre toutes les phases, y compris celles dont le code a
déjà été copié. Elle reste un support d'apprentissage : elle donne le contexte et
le squelette nécessaires, mais ne révèle pas la solution des composants que
l'utilisateur doit concevoir lui-même.

## Format retenu

Chaque exercice est un problème autonome au format « LeetCode appliqué à un
projet ». Il contient obligatoirement les sections suivantes :

1. `Difficulté` : facile, moyen ou difficile.
2. `Où travailler` : chemin exact du fichier local, emplacement et source à
   consulter lorsqu'il s'agit d'une copie.
3. `Contexte` : état du projet avant l'exercice, dépendances déjà disponibles et
   rôle du composant dans le flux complet.
4. `Objectif` : résultat précis à produire, formulé sans jargon implicite.
5. `Entrées` : nom, type, forme et signification de chaque donnée reçue.
6. `Sorties` : type, forme et signification du résultat observable.
7. `Comportement attendu` : opérations attendues dans leur ordre logique.
8. `Exemple` : code Python concret, valeurs ou formes et résultat attendu.
9. `Contraintes` : obligations, interdictions et adaptations autorisées.
10. `Code de départ` : imports, symboles voisins et squelette exact à compléter.
11. `Étapes conseillées` : progression courte sans fournir la solution finale.
12. `Vérification manuelle` : commande ou script court, sans suite de tests.
13. `Définition de terminé` : checklist factuelle.
14. `Indices` : trois niveaux d'aide, du rappel conceptuel à une indication
    presque algorithmique.

## Adaptation au travail dans un dépôt

Contrairement à un exercice algorithmique isolé, chaque problème explique ce
qui existe avant son exécution et quels exercices ultérieurs consommeront son
résultat. Les imports et signatures emploient exclusivement des symboles définis
dans la roadmap ou déjà présents dans le projet.

Les chemins sont relatifs à `eb_jepa_cifar10_comparison/`. Une section au début
du document explique cette racine une seule fois, mais chaque exercice répète le
chemin du fichier afin de rester lisible indépendamment.

Lorsqu'un symbole est placé dans un mauvais module par la roadmap actuelle, la
réécriture corrige la responsabilité du fichier. En particulier :

- les têtes restent dans `src/comparison/heads.py` ;
- `ImageSSL`, `build_model` et le comptage des paramètres vivent dans
  `src/comparison/model.py` ;
- la boucle d'entraînement ne contient aucun branchement architectural.

## Exercices de copie et exercices de conception

Un exercice de copie précise le fichier source, les symboles exacts à reprendre,
les lignes ou la structure à reconnaître, et la liste exhaustive des adaptations
autorisées. Son code de départ prépare le fichier local et ses imports sans
recopier le corps demandé.

Un exercice de conception explique les responsabilités et le flux des données,
puis fournit une signature et un squelette utilisables. Les étapes et indices
aident à construire la solution sans livrer directement son corps complet.

Les phases déjà terminées conservent leurs exercices : leur définition de
terminé permet de comparer le code présent avec le résultat attendu.

## Exemples et vérifications

Chaque exemple doit être exécutable ou explicitement présenté comme un exemple
de formes. Les tenseurs utilisent de petits batches afin que les contrôles
puissent s'exécuter localement sans A100.

Conformément au choix de l'utilisateur, aucun exercice ne demande de créer un
dossier `tests/`, d'installer `pytest` ou d'écrire un test automatisé. La
vérification se fait avec des commandes ponctuelles telles que `python -c`, un
court script interactif ou l'exécution d'un module existant.

## Niveau pédagogique

Le lecteur est supposé connaître les bases de Python, mais pas la structure de
ce dépôt, OmegaConf, VICReg, LARS ni les conventions de formes de ce modèle.
Chaque terme propre au projet est défini lors de sa première utilisation, puis
rappelé brièvement lorsqu'une confusion aurait un impact sur l'implémentation.

Les exercices ne contiennent pas de consignes vagues comme « adapte la classe »,
« gère les erreurs » ou « vérifie les invariants » sans énumérer précisément les
modifications, erreurs et invariants concernés.

## Critères d'acceptation de la roadmap

La réécriture est terminée lorsque :

- toutes les phases et tous les exercices utilisent le nouveau gabarit ;
- chaque exercice indique un fichier local exact ;
- aucun squelette ne référence un symbole inexpliqué ;
- les exemples indiquent les entrées, sorties et formes attendues ;
- toutes les adaptations de code copié sont énumérées ;
- aucune création de tests automatisés n'est demandée ;
- les responsabilités des fichiers correspondent à l'arborescence finale ;
- une recherche ne trouve aucun placeholder pédagogique tel que `TBD` ou une
  instruction non actionnable telle que « fais les adaptations nécessaires ».
