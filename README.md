[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)
<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2022 CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult the listed source code repository logs.

## Introduction

Arcane est une plateforme de développement pour les codes de calcul parallèles non structurés 2D ou 3D.

### Documentation

La documentation en ligne est accessible depuis internet :
- La documentation utilisateur se trouve ici : [Documentation utilisateur](https://arcaneframework.github.io/arcane/userdoc/html/index.html)
- La documentation développeur se trouve ici : [Documentation développeur](https://arcaneframework.github.io/arcane/devdoc/html/index.html)
- Le dépôt GitHub où est générée et stockée la documentation : [Dépôt GitHub](https://github.com/arcaneframework/arcaneframework.github.io)

### Changelog

Les dernières modifications sont dans le fichier suivant: [Changelog](arcane/doc/changelog.md)

## Compilation

Ce dépôt permet de compiler directement Arcane et ses dépendances
(Arrcon, Axlstar et Arccore)

La compilation doit se faire dans un répertoire différent de celui
contenant les sources.

Pour les prérequis, voir les répertoires [Arcane](arcane/README.md) et [Arccore](arccore/README.md):

- [Linux](#linux)

Pour récuperer les sources:

~~~{sh}
git clone --recurse-submodules /path/to/git
~~~

ou

~~~{sh}
git clone /path/to/git
cd framework && git submodule update --init --recursive
~~~

Il existe deux modes de compilations:
1. soit on compile Arcane et les projets associées (Arccon, Axlstar et
   Arccore) en même temps
2. soit on ne compile qu'un seul composant.

Le défaut est de tout compiler. La variable cmake
`FRAMEWORK_BUILD_COMPONENT` permet de choisir le mode de
compilation. Par défaut, la valeur est `all` et cela signifie qu'on
compile tout. Si la valeur est `arcane`, `arccon`, `arccore` ou
`axlstar` alors on ne compile que ces derniers. Il faut donc dans ce
cas que les dépendences éventuelles soient déjà installées (par
exemple pour Arcane il faut que Arccore soit déjà installé et
spécifier son chemin via CMAKE_PREFIX_PATH par exemple).

Pour compiler Arcane et les composantes dont il dépend (arccore, axlstar, arccon)::

~~~{sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build
cmake --build /path/to/build
~~~

A noter que dans ce mode où on compile tout à la fois le fichier
`ArcaneTargets.cmake` définira des cibles pour les packages trouvés
par Arccon (par exemple la Glib ou MPI). Cela peut poser problème si
on mélange cette installation avec une autre qui exporte les même
cibles. Pour éviter cela, il est possible de mettre à `TRUE` la variable
CMake `FRAMEWORK_NO_EXPORT_PACKAGES`.

Pour compiler uniquement Arcane en considérant que les dépendances
Arccore, Arccon, Axlstar et ArcDependencies sont déjà installées:

~~~{sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build -DFRAMEWORK_BUILD_COMPONENT=arcane -DArccon_ROOT=... -DArccore_ROOT=... -DAxlstar_ROOT=... -DArcDependencies_ROOT=...
cmake --build /path/to/build
~~~

## Linux

Cette section indique comment installer sour Linux x64 les dépendances
nécessaires.

### CMake

Il faut au moins la version 3.18 de CMake. Si elle n'est pas présente sur votre système, la commande
suivante permet de l'installer dans `/usr/local`. Il faudra ensuite
ajouter le chemin correspondant dans la variable d'environnement PATH;

~~~{sh}
# Install CMake 21.3 in /usr/local/cmake
MY_CMAKE_INSTALL_PATH=/usr/local/cmake-3.21.3
wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.tar.gz
sudo mkdir ${MY_CMAKE_INSTALL_PATH}
sudo tar -C ${MY_CMAKE_INSTALL_PATH} -x --strip-components 1 -f cmake-3.21.3-linux-x86_64.tar.gz
export PATH=${MY_CMAKE_INSTALL_PATH}/bin:${PATH}
cmake --version
~~~

### Environnement `.Net`

L'environnement `.Net` est accessible via `apt` mais vous pouvez aussi
directement télécharger un fichier `tar` contenant le binaire et les
fichiers nécessaires. Pour l'architecture `x64`, les commandes
suivantes installent l'environnement dans le répertoire `$HOME/dotnet`.

~~~{sh}
wget https://download.visualstudio.microsoft.com/download/pr/5e689959-c361-447f-af43-7c9da2595cc6/7cbba6786fe0c132d3c5a6053c85f80b/dotnet-sdk-6.0.103-linux-x64.tar.gz
mkdir -p $HOME/dotnet && tar zxf dotnet-sdk-6.0.103-linux-x64.tar.gz -C $HOME/dotnet
export DOTNET_ROOT=$HOME/dotnet
export PATH=$PATH:$HOME/dotnet
~~~

Pour d'autres architectures, la page [Download
.Net](https://dotnet.microsoft.com/en-us/download) contient la liste
des téléchargements disponibles.

### Ubuntu 20.04 via les packages systèmes

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt-get update
sudo apt-get install -y apt-utils build-essential iputils-ping python3 git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev libparmetis-dev wget
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y apt-transport-https dotnet-sdk-6.0
~~~
