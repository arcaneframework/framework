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

- [CMake 3.18+](https://cmake.org) (3.21+ pour les plateformes Windows)
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

La compilation d'Arcane nécessite d'avoir une version de [CMake](https://cmake.org) supérieure à `3.13`.
La compilation se fait obligatoirement dans un
répertoire distinct de celui des sources. On note `${SOURCE_DIR}` ce
répertoire contenant les sources et `${BUILD_DIR}` le répertoire de compilation.


Arcane dépend de `Arccon`, `Arccore` et `Axlstar`. Il est nécessaire d'avoir accès
aux sources de ces produits pour compiler. Les variables CMake
`Arccon_ROOT`, `Arccore_ROOT` et `Axlstar_ROOT doivent respectivement pointer vers les sources de ces produits.

Si `${INSTALL_PATH}` est le répertoire d'installation, les commandes suivantes permettent de compiler et installer Arcane

~~~{.sh}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -S ${SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} -DArccon_ROOT:PATH=... -DAxlstar_ROOT:PATH=... -DArccore_ROOT:PATH=...
cmake --build ${BUILD_DIR}
cmake --build ${BUILD_DIR} --target install
~~~

Par défaut, l'installation se fait dans `/usr/local` si l'option `CMAKE_INSTALL_PREFIX` n'est
pas spécifié.

Par défaut, tous les packages optionnels sont détectés
automatiquement. Il est possible de supprimer ce comportement et de
supprimer la détection automatique des packages en ajoutant
`-DARCANE_NO_DEFAULT_PACKAGE=TRUE` à la ligne de commande. Dans ce
cas, il faut préciser explicitement les packages qu'on souhaite avoir
en les spécifiant à la variable `ARCANE_REQUIRED_PACKAGE_LIST` sous
forme de liste. Par exemple, si on souhaite avoir uniquement `HDF5` et
`LibUnwind` de disponible, il faut utilise CMake comme suit:

~~~{.sh}
cmake -DARCANE_NO_DEFAULT_PACKAGE=TRUE -DARCANE_REQUIRED_PACKAGE_LIST="LibUnwind;HDF5"
~~~

Dans
### Génération de la documentation

La génération de la documentation n'a été testée que sur les plateforme Linux.
Elle nécessite l'outil [Doxygen](https://www.doxygen.nl/index.html).
L'outil Doxygen a besoin d'une installation de
[LaTeX](https://www.latex-project.org/) pour générer correctement
certaines équations. Suivant les plateformes, il peut être nécessaire
d'installer des packages LaTeX supplémentaires (par exemple pour
Ubuntu, le pakckage `texlive-latex-extra` est nécessaire).
Une fois la configuration terminée, il suffit de lancer:

Pour la documentation utilisateur:

~~~{.sh}
cmake --build ${BUILD_DIR} --target userdoc
~~~

Pour la documentation développeur

~~~{.sh}
cmake --build ${BUILD_DIR} --target devdoc
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
