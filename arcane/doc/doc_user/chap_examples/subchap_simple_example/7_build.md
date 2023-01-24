# Compilation et lancement {#arcanedoc_examples_simple_example_build}

[TOC]

Pour compiler notre application HelloWorld, nous allons donc
utiliser CMake.

Voici la liste des commandes à compléter pour compiler
et lancer notre application :

```sh
ARCANE_INSTALL_PATH=#A compléter

HW_SOURCE_DIR=#A compléter
HW_BUILD_DIR=#A compléter
HW_BUILD_TYPE=Release
HW_EXE=${HW_BUILD_DIR}/HelloWorld
HW_ARC=${HW_SOURCE_DIR}/HelloWorld.arc

cd ${HW_BUILD_DIR}

cmake \
  -S ${HW_SOURCE_DIR} \
  -B ${HW_BUILD_DIR} \
  -DCMAKE_PREFIX_PATH=${ARCANE_INSTALL_PATH} \
  -DCMAKE_BUILD_TYPE=${HW_BUILD_TYPE}

cmake --build ${HW_BUILD_DIR}

${HW_EXE} ${HW_ARC}
```
Au début de cette liste, nous définissons des variables
pour rendre le tout plus lisible et plus facilement
modifiable.

```sh
ARCANE_INSTALL_PATH=# A compléter
```
Cette ligne nous permet de définir le répertoire d'installation
de %Arcane.
\note
Exemple de ligne :
```sh
ARCANE_INSTALL_PATH=~/install_arcane
```

____

```sh
HW_SOURCE_DIR=# A compléter
```
Cette ligne nous permet de définir le répertoire contenant les sources
de notre HelloWorld.
\note
Exemple de ligne :
```sh
HW_SOURCE_DIR=~/src_helloworld
```

____

```sh
HW_BUILD_DIR=# A compléter
```
Cette ligne nous permet de définir le répertoire où sera compilé notre application.
\note
Exemple de ligne :
```sh
HW_BUILD_DIR=~/build_hw
```

____

```sh
HW_BUILD_TYPE=Release
```
Cette ligne nous permet de définir le type de build
que l'on souhaite.
On a le choix entre `Debug`, `Check` et `Release`.
`Debug` nous permet d'avoir des informations supplémentaire
pour pouvoir débugger avec un débuggeur.
`Check` permet de rajouter des vérifications supplémentaires
pour éviter les problèmes de dépassement de tableau par exemple.

\note `Debug` inclut automatiquement `Check`.

____

```sh
HW_EXE=${HW_BUILD_DIR}/HelloWorld
HW_ARC=${HW_SOURCE_DIR}/HelloWorld.arc

cd ${HW_BUILD_DIR}
```
À l'aide des informations que nous avons donné précédement,
on peut déduire l'emplacement de l'exécutable (`HW_EXE`)
et l'emplacement de notre jeu de données (`HW_ARC`).

Personnellement, je préfère me situer dans le dossier de build
pour les commandes ci-après, donc j'effectue un `cd ${HW_BUILD_DIR}`,
mais ça reste facultatif.

____

```sh
cmake \
  -S ${HW_SOURCE_DIR} \
  -B ${HW_BUILD_DIR} \
  -DCMAKE_PREFIX_PATH=${ARCANE_INSTALL_PATH} \
  -DCMAKE_BUILD_TYPE=${HW_BUILD_TYPE}
```
On demande à CMake de configurer le dossier de build
grâce à/aux CMakeLists.txt pour que Make puisse compiler
notre projet. Cette commande ne modifiera pas le répertoire
contenant les sources de notre projet (c'est le cas pour toutes les
commandes présentées ici d'ailleurs).

____

```sh
cmake --build ${HW_BUILD_DIR}
```
On demande à CMake d'appeler Make pour compiler notre projet.
On pourrait utiliser directement la commande `make` mais
si un jour, on veut utiliser un autre programme que Make
(comme Ninja), il faudra changer cette commande. Alors que là, 
c'est CMake qui gère selon la configuration faite précédemment.

____

```sh
${HW_EXE} ${HW_ARC}
```
Enfin, on peut lancer notre HelloWorld !
On doit aussi préciser la position de notre jeu de données.
Si l'on a plusieurs jeux de données, il suffit de changer
la variable `${HW_ARC}` et de relancer HelloWorld sans
avoir besoin de recompiler.

____

Voilà pour ce sous-chapitre dédié à la construction d'un hello world avec
%Arcane. Les bases sont présentées ici. Néanmoins, pour aller plus loin,
il est conseillé de lire tous les chapitres de cette documentation.

Si un problème est présent dans ce sous-chapitre, vous pouvez ouvrir une
issue dans le GitHub d'%Arcane.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_cmake
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example
</span>
</div>
