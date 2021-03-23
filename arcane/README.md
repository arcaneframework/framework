[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)

# Arcane

<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2021- CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult the listed source code repository logs.

## Introduction

Arcane est une platforme de développement pour les codes de calcul parallèles non structurés 2D ou 3D.

## Compilation et installation

### Pré-requis

Un compilateur supportant le C++17:

- GCC 7+
- Clang 6+
- Visual Studio 2019 (version 16.8+)

Les outils et bibliothèques suivants sont requis:

- [CMake 3.13+](https://cmake.org)
- [.Net Core 3.0+](https://dotnet.microsoft.com/download)
- [GLib](https://www.gtk.org/)
- [LibXml2](http://www.xmlsoft.org/)

Les outils et bibliothèques suivants sont optionnels mais fortement recommandés:

- IntelTBB 2018+
- MPI (implémentation MPI 3.1 nécessaire)

Les outils et bibliothèques suivants sont optionnels:

- [Swig 4.0+](http://swig.org/)
- [HDF5 1.10+](https://www.hdfgroup.org/solutions/hdf5/)

### Compilation

La compilation d'Arcane nécessite d'avoir une version de [CMake](https://cmake.org) supérieure à `3.13`. La compilation se fait obligatoirement dans un
répertoire distinct de celui des sources. On note `${SOURCE_DIR}` ce
répertoire contenant les sources et `${BUILD_DIR}` le répertoire de compilation.


Arcane dépend de `Arccon`, `Arccore` et `Axlstar`. Il est nécessaire d'avoir accès
aux sources de ces produits pour compiler. Les variables CMake
`Arccon_ROOT`, `Arccore_ROOT` et `Axlstar_ROOT doivent respectivement pointer vers les sources de ces produits.

Si `${INSTALL_PATH}` est le répertoire d'installation, les commandes suivantes permettent de compiler et installer Arcane

~~~{.sh}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -S ${SOURCE_DIR} -DCMAKE_PREFIX_PATH=${INSTALL_PATH} -DArccon_ROOT:PATH=... -DAxlstar_ROOT:PATH=... -DArccore_ROOT:PATH=...
cmake --build .
cmake --build . --target install
~~~

Par défaut, l'installation se fait dans /usr/local si l'option `CMAKE_PREFIX_PATH` n'est
pas spécifié.

### Génération de la documentation

La génération de la documentation n'a été testée que sur les plateforme Linux.
Elle nécessite l'outil [Doxygen](https://www.doxygen.nl/index.html).
Une fois la configuration terminée, il suffit de lancer:

Pour la documentation utilisateur:

~~~{.sh}
cmake --build . --target userdoc
~~~

Pour la documentation développeur

~~~{.sh}
cmake --build . --target devdoc
~~~

La documentation utilisateur ne contient les infos que des classes
utiles pour le développeur.

### Compilation et tests des exemples

Une fois Arcane installé dans `${INSTALL_PATH}`, il est possible de compiler les exemples:

~~~{.sh}
# Recopie les exemples dans /tmp/samples
cp -r ${INSTALL_PATH}/samples /tmp
cd /tmp/samples
cmake .
cmake --build .
ctest
~~~

La commande `ctest` permet de lancer les tests sur les exemples.