[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)

# Arccore

<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2023 CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult the listed source code repository logs.

## Introduction

Ce dépôt contient les sources de **Arccore**.

### Pré-requis

Un compilateur supportant le C++17:

- GCC 7+
- Clang 6+
- Visual Studio 2019 (version 16.8+)

Les outils et bibliothèques suivants sont optionnels mais fortement recommandés:

- MPI (implémentation MPI 3.1 nécessaire)

### Compilation

La compilation de Arccore nécessite d'avoir une version de
 [CMake](https://cmake.org) supérieure à `3.13`. La compilation se
 fait obligatoirement dans un répertoire distinct de celui des
 sources. On note `${SOURCE_DIR}` le répertoire contenant les sources
 et `${BUILD_DIR}` le répertoire de compilation.

~~~{.sh}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -S ${SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} ...
cmake --build ${BUILD_DIR}
cmake --build ${BUILD_DIR} --target install
~~~

Par défaut, l'installation se fait dans `/usr/local` si l'option `CMAKE_INSTALL_PREFIX` n'est
pas spécifiée.

Il est possible de positioner la variable CMake `ARCCORE_BUILD_MODE`
avec l'une des valeurs suivantes:

- `Debug`: active les macros d'afficahge `debug()` et le mode
  vérification (mode 'check'). Dans ce mode, les macros
  `ARCCORE_DEBUG` et `ARCCORE_CHECK` sont définies.
- `Check`: active le mode vérification, dans lequel on vérifie
  notamment les débordements de tableau. Dans ce mode, la macro
  `ARCCORE_CHECK` est définie
- `Release`: mode sans vérification ni message de débug.

La valeur par défaut de `ARCCORE_BUILD_MODE` est `Debug` si
`CMAKE_BUILD_TYPE` vaut `Debug`. Sinon, la valeur par défaut est
`Release`.
