[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)
<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2021 CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult the listed source code repository logs.

## Introduction

Arcane est une platforme de développement pour les codes de calcul parallèles non structurés 2D ou 3D.

## Compilation

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
