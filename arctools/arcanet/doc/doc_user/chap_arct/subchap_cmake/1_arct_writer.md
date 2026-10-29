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
une variable CMake puis, à la fin, d'écrire son contenu dans un fichier ; l'autre ouvre, lit, écrit et ferme un fichier
ARCT à chaque étape.

Le premier mode permet d'optimiser un peu les performances mais nécessite d'avoir accès à la variable CMake en lecture
et en écriture à chaque appel aux fonctions CMake ARCT (donc pas de fonctions utilisateur, uniquement des macros).

Le second est plus simple mais peut devenir lent pour les gros fichiers ARCT. De plus, il nécessite de donner
l'emplacement du fichier ARCT à chaque fois (si celui par défaut n'est pas utilisé).

Dans tous les exemples de cette documentation, on va utiliser le second mode. On va donc présenter le premier mode
maintenant.

Ce mode nécessite l'utilisation de deux fonctions. Entre ces deux fonctions, on peut décrire le fichier ARCT avec les
fonctions présenté plus tard.

--------------------------

### arct_begin

```cmake
arct_begin(OVERWRITE_ARCT ARCT_PATH <path>)
```

Cette première fonction permet d'ouvrir le fichier ARCT et de le copier dans une variable CMake s'il existe. Si ce
fichier n'existe pas, la variable aura simplement un objet JSON vide.

Une option est disponible :

- `OVERWRITE_ARCT` : permet d'écraser le fichier ARCT existant s'il existe.

Un argument `oneValueArgs` facultatif est disponible :

- `ARCT_PATH <path>` :  permet de donner l'emplacement du fichier ARCT à écrire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`.

--------------------------

### arct_end

```cmake
arct_end()
```

Cette fonction permet simplement d'écrire le fichier ARCT à partir de la variable CMake.

--------------------------

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

Les fonctions CMake ARCT entre les fonctions `arct_config_create_or_edit_XXX()` et `arct_config_create_or_edit_end()`
doivent pouvoir modifier les variables CMake définies par `arct_config_create_or_edit_XXX()`.

Les fonctions `arct_config_create_or_edit_XXX()` permettant de se substituer de la fonction `arct_begin()`, ces
fonctions reprennent donc les paramètres de `arct_begin()`.

Ces fonctions permettent de créer une partie de fichier ARCT (si celle-ci n'existe pas déjà) et de pouvoir la modifier.
Toutes les fonctions CMake ARCT entre `arct_config_create_or_edit_XXX()` et `arct_config_create_or_edit_end()`
concerneront le bloc `XXX`.

--------------------------

### arct_config_create_or_edit_general

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT ARCT_PATH <path>)
```

Cette fonction permet de créer et/ou de modifier le bloc de configuration `general` du fichier ARCT :

```json5
{
  "general": {
    // <config_block>
  }
}
```

Une option est disponible :

- `OVERWRITE_ARCT` : permet d'écraser le fichier ARCT existant s'il existe. Aucun effet si le premier mode est utilisé.

Un argument `oneValueArgs` facultatif est disponible :

- `ARCT_PATH <path>` :  permet de donner l'emplacement du fichier ARCT à écrire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`. Aucun effet si le premier mode est utilisé.

--------------------------

### arct_config_create_or_edit_cases

```cmake
arct_config_create_or_edit_cases(OVERWRITE_ARCT ARCT_PATH <path> NAME_CASE <name_case>)
```

Cette fonction permet de créer et/ou de modifier un des blocs de configuration de la partie `cases` du fichier ARCT :

```json5
{
  "cases": {
    "<name_cas>": {
      // <config_block>
    }
  }
}
```

Une option est disponible :

- `OVERWRITE_ARCT` : permet d'écraser le fichier ARCT existant s'il existe. Aucun effet si le premier mode est utilisé.

Deux arguments `oneValueArgs` sont disponibles :

- `ARCT_PATH <path>` (facultatif) :  permet de donner l'emplacement du fichier ARCT à écrire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`. Aucun effet si le premier mode est utilisé.
- `NAME_CASE <name_case>` (**obligatoire**) : permet de donner le nom du cas de base à créer/modifier par la suite.

--------------------------

### arct_config_create_or_edit_commons

```cmake
arct_config_create_or_edit_commons(OVERWRITE_ARCT ARCT_PATH <path> NAME_COMMON <name_common>)
```

Cette fonction permet de créer et/ou de modifier un des blocs de configuration de la partie `commons` du fichier ARCT :

```json5
{
  "commons": {
    "<name_common>": {
      // <config_block>
    }
  }
}
```

Une option est disponible :

- `OVERWRITE_ARCT` : permet d'écraser le fichier ARCT existant s'il existe. Aucun effet si le premier mode est utilisé.

Deux arguments `oneValueArgs` sont disponibles :

- `ARCT_PATH <path>` (facultatif) :  permet de donner l'emplacement du fichier ARCT à écrire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`. Aucun effet si le premier mode est utilisé.
- `NAME_COMMON <name_common>` (**obligatoire**) : permet de donner le nom de la dépendance à créer/modifier par la
  suite.

--------------------------

### arct_config_create_or_edit_variation

```cmake
arct_config_create_or_edit_variation(OVERWRITE_ARCT ARCT_PATH <path> NAME_VARIATION <name_variation> NAME_VARIANT <name_variant>)
```

Cette fonction permet de créer et/ou de modifier un des blocs de configuration d'un variant d'une variation de la partie
`variations` du fichier ARCT :

```json5
{
  "variations": {
    "<name_variation>": {
      "<name_variant>": {
        // <config_block>
      }
    }
  }
}
```

Une option est disponible :

- `OVERWRITE_ARCT` : permet d'écraser le fichier ARCT existant s'il existe. Aucun effet si le premier mode est utilisé.

Trois arguments `oneValueArgs` sont disponibles :

- `ARCT_PATH <path>` (facultatif) :  permet de donner l'emplacement du fichier ARCT à écrire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`. Aucun effet si le premier mode est utilisé.
- `NAME_VARIATION <name_variation>` (**obligatoire**) : permet de donner le nom de la variation à créer/modifier par la
  suite.
- `NAME_VARIANT <name_variant>` (**obligatoire**) : permet de donner le nom du variant à créer/modifier par la suite.

--------------------------

### arct_config_create_or_edit_end

```cmake
arct_config_create_or_edit_end()
```

Cette fonction permet de mettre à jour la variable contenant le JSON ARCT. De plus, dans le second mode, elle écrit la
variable dans le fichier ARCT.

--------------------------

## Modification des propriétés du bloc réservé ARCT

Plusieurs fonctions sont disponibles pour modifier les propriétés du bloc réservé ARCT :

```json5
{
  "XXX": {
    "_": {
      // <reserved_block>
    }
  }
}
```

*`XXX` correspond au type de bloc choisi lors de l'appel à `arct_config_create_or_edit_XXX()`.*


--------------------------

### arct_config_define_name

```cmake
arct_config_define_name(NAME <name>)
```

Cette fonction permet d'ajouter un nom pour un bloc de configuration. Si un nom existe déjà, il sera écrasé :

```json5
{
  "XXX": {
    "_": {
      "name": "<name>"
    }
  }
}
```

\warning Il s'agit d'un nom d'affichage : le nom du bloc de configurations ne change pas.

Ce nom peut donc contenir tous les symboles que vous souhaitez.

Un argument `oneValueArgs` est disponible :

- `NAME <name>` (**obligatoire**) : permet de donner un nom d'affichage au bloc de configuration.

--------------------------

### arct_config_dependency

```cmake
arct_config_dependency([BEFORE|AFTER] ADD <name_dep>)
```

Cette fonction permet d'ajouter une dépendance à un bloc de configuration.

Par défaut, l'option `BEFORE` est sélectionnée.

Si l'option `BEFORE` est sélectionnée :

```json5
{
  "XXX": {
    "_": {
      "depend_b": [..., "<name_dep>"]
    }
  }
}
```

Si l'option `AFTER` est sélectionnée :

```json5
{
  "XXX": {
    "_": {
      "depend_a": [..., "<name_dep>"]
    }
  }
}
```

Deux options sont disponibles :

- `BEFORE` : permet d'ajouter une dépendance *before*. Option par défaut.
- `AFTER` : permet d'ajouter une dépendance *after*.

Un argument `oneValueArgs` est disponible :

- `ADD <name_dep>` (**obligatoire**) : permet d'ajouter une dépendance à la fin du tableau des dépendances.

--------------------------

### arct_config_variation

```cmake
arct_config_variation([BEFORE|AFTER] [ADD <name_variation>|APPEND <name_variation>] [EQUAL <name_variant>|INCLUDE <name_variant>...|EXCLUDE <name_variant>...])
```

Cette fonction permet d'ajouter ou de modifier une dépendance de type variation.

Par défaut, l'option `BEFORE` est sélectionnée.

Si l'option `BEFORE` est sélectionnée :

```json5
{
  "XXX": {
    "_": {
      "depend_b": [..., "<name_variation><equal_include_exclude><name_variant>"]
    }
  }
}
```

Si l'option `AFTER` est sélectionnée :

```json5
{
  "XXX": {
    "_": {
      "depend_a": [..., "<name_variation><equal_include_exclude><name_variant>"]
    }
  }
}
```

L'argument `ADD` permet d'ajouter une variation dans la liste des dépendances. L'argument `APPEND` permet d'ajouter un
limiteur de variant à une dépendance déjà ajoutée (si la liste contient `nb_proc!1`, `APPEND nb_proc EXCLUDE 4`
donnerait `nb_proc!1!4`).

Deux options sont disponibles :

- `BEFORE` : permet d'ajouter une dépendance *before*. Option par défaut.
- `AFTER` : permet d'ajouter une dépendance *after*.

Trois arguments `oneValueArgs` sont disponibles :

- `ADD <name_variation>` (**obligatoire ou APPEND**) : permet d'ajouter une variation à la fin du tableau des
  dépendances.
- `APPEND <name_variation>` (**obligatoire ou ADD**) : permet d'ajouter une ou plusieurs inclusions/exclusions de
  variants dans une variation déjà présente dans le tableau des dépendances.
- `EQUAL <name_variant>` (facultatif) : permet de résoudre la variation passée avec l'argument `ADD` ou `APPEND`
  (symbole `=`).

Deux arguments `multiValueArgs` sont disponibles :

- `INCLUDE <name_variant>...` (facultatif) : permet de résoudre partiellement la variation passée avec l'argument `ADD`
  ou `APPEND` (symbole `+`).
- `EXCLUDE <name_variant>...` (facultatif) : permet de résoudre partiellement la variation passée avec l'argument `ADD`
  ou `APPEND` (symbole `!`).

## Modification des propriétés du bloc réservé Arcane

Plusieurs fonctions sont disponibles pour modifier les propriétés du bloc réservé Arcane :

```json5
{
  "XXX": {
    "arcane": {
      // <reserved_block>
    }
  }
}
```

*`XXX` correspond au type de bloc choisi lors de l'appel à `arct_config_create_or_edit_XXX()`.*

--------------------------

### arct_arcane_dataset

```cmake
arct_arcane_dataset(PATH <path_dataset>)
```

Cette fonction permet de définir un jeu de données pour Arcane. Si un chemin existe déjà, il sera écrasé :

```json5
{
  "XXX": {
    "arcane": {
      "dataset": "<path_dataset>"
    }
  }
}
```

Un argument `oneValueArgs` est disponible :

- `PATH <path_dataset>` (**obligatoire**) : permet de donner un chemin pour trouver le jeu de données. Le chemin peut
  être relatif au répertoire de travail.

--------------------------

### arct_arcane_option

```cmake
arct_arcane_option(ADD <name_option> VALUE <value_option>)
```

Cette fonction permet d'ajouter une option dans la liste des options à donner à Arcane :

```json5
{
  "XXX": {
    "arcane": {
      "options": {
        ...,
        "<name_option>": "<value_option>"
      }
    }
  }
}
```

Une option Arcane classique se passe par la ligne de commande. Par exemple : `-A,MaxIteration=10` va donner ici
`arct_arcane_option(ADD "MaxIteration" VALUE "10")`.

Compatible avec toutes les options Arcane (même les modifications du jeu de données, exemple :
`arct_arcane_option(ADD "//meshes/mesh[]/generator/nb-part-x" VALUE "4")`).

Deux arguments `oneValueArgs` sont disponibles :

- `ADD <name_option>` (**obligatoire**) : permet de donner le nom de l'option à ajouter.
- `VALUE <value_option>` (**obligatoire**) : permet de donner la valeur de l'option à ajouter.

## Modification des propriétés du bloc réservé ArcCher

Plusieurs fonctions sont disponibles pour modifier les propriétés du bloc réservé ArcCher (Arcane launCher) :

```json5
{
  "XXX": {
    "arccher": {
      // <reserved_block>
    }
  }
}
```

*`XXX` correspond au type de bloc choisi lors de l'appel à `arct_config_create_or_edit_XXX()`.*

--------------------------

### arct_arccher_mpi

```cmake
arct_arccher_mpi(PATH <path_mpirun> NB_RANK <nb_rank_mpi>)
```

Cette fonction permet de définir les options MPI :

```json5
{
  "XXX": {
    "arccher": {
      "mpi_binary": "<path_mpirun>",
      "mpi": "<nb_rank_mpi>"
    }
  }
}
```

Deux actions peuvent être faites, soit en un seul appel, soit en deux :

```cmake
arct_arccher_mpi(PATH /usr/bin/mpirun NB_RANK 4)
```

est équivalent à :

```cmake
arct_arccher_mpi(PATH /usr/bin/mpirun)
arct_arccher_mpi(NB_RANK 4)
```

Deux arguments `oneValueArgs` sont disponibles :

- `PATH <path_mpirun>` (facultatif) : permet de donner un chemin pour le binaire `mpirun/mpiexec`.
- `NB_RANK <nb_rank_mpi>` (facultatif) : permet de donner un nombre de processus à utiliser.

--------------------------

### arct_arccher_envvar

```cmake
arct_arccher_envvar(ADD <name_envvar> VALUE <value_envvar>)
```

Cette fonction permet de définir des variables d'environnement qui seront définies au lancement :

```json5
{
  "XXX": {
    "arccher": {
      "env_var": {
        "<name_envvar>": "<value_envvar>"
      }
    }
  }
}
```

Deux arguments `oneValueArgs` sont disponibles :

- `ADD <name_envvar>` (**obligatoire**) : permet de donner le nom de la variable d'environnement à ajouter.
- `VALUE <value_envvar>` (**obligatoire**) : permet de donner la valeur de la variable d'environnement à ajouter.

--------------------------

### arct_arccher_executable

```cmake
arct_arccher_executable(PATH <path_executable>)
```

Cette fonction permet de définir où se situe l'exécutable du programme à lancer par ArcCher :

```json5
{
  "XXX": {
    "arccher": {
      "code_binary": "<path_executable>"
    }
  }
}
```

Un argument `oneValueArgs` est disponible :

- `PATH <path_executable>` (facultatif) : permet de donner un chemin pour l'exécutable à lancer.

--------------------------

### arct_arccher_working_dir

```cmake
arct_arccher_working_dir(PATH <path_workingdir>)
```

Cette fonction permet de définir où est le répertoire de travail. Ça permet d'utiliser des chemins relatifs pour
l'exécutable, le jeu de données, les maillages du jeu de données, &c.

```json5
{
  "XXX": {
    "arccher": {
      "working_dir": "<path_workingdir>"
    }
  }
}
```

Un argument `oneValueArgs` est disponible :

- `PATH <path_workingdir>` (facultatif) : permet de donner le chemin du répertoire de travail.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_arct_to_ctest
</span>
</div>
