[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)
<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2023 CEA/IFPEN. All rights reserved.

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

Les dernières modifications sont dans le fichier suivant: [Changelog](arcane/doc/doc_common/changelog.md)

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

Pour compiler uniquement Arcane en considérant que les dépendances
Arccore, Arccon, Axlstar et ArcDependencies sont déjà installées:

~~~{sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build -DFRAMEWORK_BUILD_COMPONENT=arcane -DArccon_ROOT=... -DArccore_ROOT=... -DAxlstar_ROOT=... -DArcDependencies_ROOT=...
cmake --build /path/to/build
~~~

## Linux

Cette section indique comment installer sous Linux x64 les dépendances
nécessaires.

### CMake

Il faut au moins la version 3.21 de CMake. Si elle n'est pas présente sur votre système, la commande
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

Vous pouvez aussi installer directement CMake via [snap](https://snapcraft.io/):
~~~{sh}
sudo snap install --classic cmake
~~~


### Environnement `.Net`

L'environnement `.Net` est accessible via `apt` mais vous pouvez aussi
directement télécharger un fichier `tar` contenant le binaire et les
fichiers nécessaires. Pour l'architecture `x64`, les commandes
suivantes installent l'environnement dans le répertoire `$HOME/dotnet`.

~~~{sh}
wget https://download.visualstudio.microsoft.com/download/pr/372b11de-1321-44f3-aad7-040842babe62/c5925f9f856c3a299e97c80283317275/dotnet-sdk-6.0.304-linux-x64.tar.gz
mkdir -p $HOME/dotnet && tar zxf dotnet-sdk-6.0.304-linux-x64.tar.gz -C $HOME/dotnet
export DOTNET_ROOT=$HOME/dotnet
export PATH=$HOME/dotnet:$PATH:
~~~

Pour d'autres architectures, la page [Download
.Net](https://dotnet.microsoft.com/en-us/download) contient la liste
des téléchargements disponibles.

### Ubuntu 20.04 via les packages systèmes

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt-get update
sudo apt-get install -y apt-utils build-essential iputils-ping python3 \
git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
libparmetis-dev wget
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y apt-transport-https dotnet-sdk-6.0
~~~

### Ubuntu 22.04 via les packages systèmes

Sur Ubuntu 22.04, les versions de CMake et de '.Net' sont suffisamment
récentes pour pouvoir être installés via les packages système.

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt-get update
sudo apt-get install -y apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
  libparmetis-dev dotnet6 cmake
~~~

Il est aussi possible d'installer les packages optionnels suivants:

~~~{sh}
# Pour google test:
sudo apt-get install -y googletest

# Pour Ninja:
sudo apt-get install -y ninja-build

# Pour le wrapper C#:
sudo apt-get install -y swig4.0

# Pour Hypre
sudo apt-get install -y libhypre-dev

# Pour PETSc
sudo apt-get install -y libpetsc-real-dev

# Pour Trilinos
sudo apt-get install -y libtrilinos-teuchos-dev libtrilinos-epetra-dev \
  libtrilinos-tpetra-dev libtrilinos-kokkos-dev libtrilinos-ifpack2-dev \
  libtrilinos-ifpack-dev libtrilinos-amesos-dev libtrilinos-galeri-dev \
  libtrilinos-xpetra-dev libtrilinos-epetraext-dev \
  libtrilinos-triutils-dev libtrilinos-thyra-dev \
  libtrilinos-kokkos-kernels-dev libtrilinos-rtop-dev \
  libtrilinos-isorropia-dev libtrilinos-belos-dev \

# Pour Zoltan
sudo apt-get install -y libtrilinos-ifpack-dev libtrilinos-anasazi-dev \
  libtrilinos-amesos2-dev libtrilinos-shards-dev libtrilinos-muelu-dev \
  libtrilinos-intrepid2-dev libtrilinos-teko-dev libtrilinos-sacado-dev \
  libtrilinos-stratimikos-dev libtrilinos-shylu-dev \
  libtrilinos-zoltan-dev libtrilinos-zoltan2-dev
~~~

### Arch Linux/Manjaro via Pacman/YAY

Les commandes suivantes permettent d'installer CMake, .Net et les dépendances
nécessaires pour Arcane (ainsi que les dépendances optionnelles `TBB`, `HDF5` et `ParMetis`):

~~~{sh}
sudo pacman -Syu
sudo pacman -S gcc cmake python git gcc-fortran glib2 libxml2 hdf5-openmpi wget tbb dotnet-sdk aspnet-runtime aspnet-targeting-pack
yay -S aur/parmetis
~~~
