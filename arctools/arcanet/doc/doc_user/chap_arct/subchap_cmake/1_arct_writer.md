# Écrivain de fichier ARCT en CMake {#arcanetdoc_arct_cmake_arct_writer}

[TOC]

Pour écrire des tests de non-régression, nous avons créé plusieurs fonctions CMake afin de construire un fichier ARCT
petit à petit. Ces fonctions peuvent remplacer les `add_test()` mais le fonctionnement est totalement différent.

`add_test()` permet d'ajouter un test en spécifiant son nom et la commande à exécuter pour lancer le test. Un test passe
si la commande renvoie 0 (par défaut).

La contribution du projet ArcaNet ajoute une étape supplémentaire : le fichier ARCT.

Au lieu de faire des `add_test()`, l'utilisateur peut utiliser les fonctions CMake fournies pour décrire ces cas tests,
ajouter des variations, &c. Avec ArcCher, il est aussi possible de lancer les tests avec MPI (et peut-être même sur
calculateur plus tard).

Une fois le fichier ARCT construit, il est aujourd'hui possible de le convertir automatiquement en CTest via une autre
fonction CMake. Plus tard, il sera possible d'utiliser ce fichier dans un autre lanceur de cas test : Schedane.

## Initialisation du fichier ARCT

Deux modes d'utilisation de ces fonctions sont disponibles ; l'un permettant d'écrire le fichier ARCT petit-à-petit dans
une variable CMake puis, à la fin, d'écrire son contenu dans un fichier ; l'autre ouvre, lit et écrit dans un fichier
ARCT directement à chaque fois.

Le premier mode permet d'optimiser un peu les performances mais nécessite d'avoir accès à la variable CMake en lecture
et en écriture à chaque appel aux fonctions CMake ARCT (donc pas de fonctions utilisateur, uniquement des macros). Le
second est plus simple mais peut devenir lent pour les gros fichiers ARCT.

Dans tous les exemples de cette documentation, on va utiliser le second mode. On va donc présenter le premier mode
maintenant.

Ce mode nécessite l'utilisation de deux fonctions. Entre ces deux fonctions, on peut décrire le fichier ARCT avec les
fonctions présenté plus tard.

----

### arct_begin

```cmake
arct_begin(OVERWRITE_ARCT ARCT_PATH <path>)
```

Cette première fonction permet d'ouvrir le fichier ARCT et de le copier dans une variable CMake s'il existe. Si ce
fichier n'existe pas, la variable aura simplement un objet JSON vide.

Une option est disponible : `OVERWRITE_ARCT`. Elle permet d'écraser le fichier ARCT existant s'il existe.

Un argument `oneValueArgs` est disponible : `ARCT_PATH <path>`. Il permet de donner l'emplacement du fichier ARCT à
écrire. La valeur par défaut est `${CMAKE_BINARY_DIR}/testlist.arct`.

----

### arct_end

```cmake
arct_end()
```

Cette fonction permet simplement d'écrire le fichier ARCT à partir de la variable CMake le contenant.

----

Dans le second mode, le fichier est ouvert/lu par les fonctions `arct_config_create_or_edit_XXX()` et est écrit par la
fonction `arct_config_create_or_edit_end()`.

Pas besoin de spécifier un mode ou l'autre : l'utilisation de `arct_begin()` suffit pour déterminer l'utilisation du
premier mode.

\warning Il n'est pas encore possible de créer des blocs réservés perso. Les seuls disponibles sont ceux d'Arcane et
d'ArcCher.

## Ouverture et fermeture de blocs de configurations

Dans un fichier ARCT, il y a quatre parties contenant des blocs de configurations :

- `general`,
- `cases`,
- `commons`,
- `variations`.

Pour chacune de ces parties, il y a une fonction permettant de créer ou éditer un bloc de configuration :

- `arct_config_create_or_edit_general()`,
- `arct_config_create_or_edit_cases()`,
- `arct_config_create_or_edit_commons()`,
- `arct_config_create_or_edit_variations()`.

Et une fonction pour terminer l'édition (et écrire le fichier ARCT dans le cas du second mode) :

- `arct_config_create_or_edit_end()`.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake
</span>
<!--<span class="next_section_button">
\ref arcanetdoc_arct_general
</span>-->
</div>
