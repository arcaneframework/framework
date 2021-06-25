[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)
Compilation
===========

Ce dépôt permet de compiler directement Arcane et ses dépendances
(Arrcon, Axlstar et Arccore)

La compilation doit se faire dans un répertoire différent de celui
contenant les sources.

Pour les prérequis, voir les répertoires [Arcane](arcane/README.md) et [Arccore](arccore/README.md).

Pour récuperer les sources:

~~~{.sh}
git clone /path/to/git
git submodule update --init --recursive
~~~

**ATTENTION: Pour des raisons de compatibilité, le fichier `CMakeLists.txt` à la
racine du projet compile uniquement Arcane**. La commande suivante permet de
configurer et compiler Arcane en supposant que les composantes
`arccon`, `arccore` et `Axlstar` sont déjà installées:

~~~{.sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build
cmake --build /path/to/build
~~~

Si on souhaite compiler l'ensemble des composantes en une seule fois,
il faut utiliser le fichier `CMakeLists.txt` qui est dans
`_common/build_all`. Par exemple:

~~~{.sh}
mkdir /path/to/build
cmake -S /path/to/sources/_common/build_all -B /path/to/build
cmake --build /path/to/build
~~~
