# README

## Historique

Pour un package donné, la recherche avec CMake se fait avec la
commande `find_package`. Cependant, le fonctionnement de cette
commande a évolué avec le temps et elle propose plusieurs mécanismes:

1. historiquement (les années 2000 à 2010), pour un package donné
   `<pkg>` il fallait fournir un fichier `Find<pkg>.cmake`. Ce fichier
   était exécuté lors d'un `find_package`  et positionnait des
   variables dont les principales étaient `<pkg>_INCLUDE_DIRS` et
   `<pkg>_LIBRARIES`. Le code pouvait ensuite utiliser ces variables
   dans les commandes telles que `target_include_directories` ou
   `target_link_libraries`. A cette époque il n'y avait pas la notion
   de transitivité sur les cibles et il fallait tout positionner à la
   main.

2. Si le package utilise lui-même CMake, alors il peut installer un
   fichier `<pkg>Config.cmake` qui sera executé lors de la commande
   `find_package`. Ce mécanisme représente le cas idéal pour
   l'utilisation de CMake. Cependant, il faut gérer son
   évolution dans le temps si le nom des cibles changent dans la
   gestion du package par exemple.

3. Le dernier mode est similaire au mode 1 car il utilise un
   fichier `Find<pkg>.cmake` mais il créé une cible interface au lieu
   de juste positionner `<pkg>_INCLUDE_DIRS` et
   `<pkg>_LIBRARIES`. Cette cible interface est directement utilisée
   via un `target_link_libraries`.

La difficulté est que pour un package donnée les 3 méthodes peuvent
coexister. Par exemple, pour HDF5, il est possible de le compiler avec
CMake. Dans ce cas un fichier `HDF5Config.cmake` sera disponible. Si
on le compile sans CMake, alors c'est le fichier `FindHDF5.cmake` qui
sera trouvé. Mais les deux mécanismes ne proposent pas le même
fonctionnement ni même le même nom de cible importé (`hdf5::hdf5` pour
CMake et `hdf5::hdf5-shared` ou `hdf5::hdf5-static` pour
HDF5Config.cmake). De même pour `LibXml2` qui peut s'installer via
CMake ou via un configure. CMake fournit un fichier
`FindLibXml2.cmake` et ce dernier utilise le mécanisme (3) mais
uniquement depuis la version 3.12 sinon il utilise le mécanisme (1).

## Fonctionnement

Arccon fournit des fonctions CMake pour gérer les packages classiques
ainsi que les cibles CMake associées.

Son objectiuf est de proposer une abstraction pour gérer les packages de
manière uniforme en fonction des versions de CMake et de la manière
dont est installé le produit afin de ne pas avoir à modifier les
fichiers CMake de l'application.

Historiquement, pour gérer tout cela de manière uniforme, Arccon
génère pour chaque package une cible interface `arcconpkg_<pkg>` qui
contient les propriétés INCLUDE_DIRS et LIBRARIES.

Ce mécanisme est longtemps resté fonctionnel au pris de la création de
fichiers `Find<pkg>.cmake` plus ou moins complexes. Cependant, il ne
fonctionne plus très bien pour plusieurs raisons:

- il peut y avoir d'autres paramètres que la liste des fichiers
  d'en-tête et la liste des noms des bibliohèques. Par exemple une
  cible peut ajouter ces propres options de compilation. Ces dernières
  ne seront pas prises en compte par Arccon avec ce mécanisme.

- le nom des bibliothèques peuvent être différentes suivant les modes
  de compilation (Debug ou Release) mais aussi contenir avec le CMake
  moderne des 'generator expression' que Arccon ne peut pas gérer.

- la transitivité des cibles n'est pas gérée non plus.

### Solution 1 (2015-2022)

La solution retenue pour éviter ce problème est d'utiliser la notion d'alias
disponible dans CMake depuis la version 3.12. Pour un package donnée
on crée une cible alias `arccon::<pkg>` comme suit:

1. si le package est géré par Arccon, alors `arccon::<pkg>` est un
   alias de `arcconpkg_<pkg>`.

2. si le package est une cible importée, alors `arccon::<pkg>` est un
   alias vers le nom de cette cible (par exemple `LibXml2::LibXml2`).

Cependant, il y a plusieurs incovénients à ce mécanisme pour le point (2)

- dans le cas des cibles importées (le point (2)) les alias ne sont
  pas globaux mais ont une portée restreinte au CMakeLists.txt qui
  fait le `find_package`.

- l'alias ne peut spécifier qu'une seule cible. Dans certains cas
  pourtant il faudrait pouvoir associer plusieurs cibles importées à
  un package. On peut prendre comme exemple le package défini dans Arcane
  `vtkIOXML`. Dans les versions de VTK antérieures à la version 9.0,
  ce package correspont à un package VTK de même nom. Mais à partir de
  la version 9.0, ce package est obsolète et il faut deux cibles
  importées `VTK::IOXML` et `VTK::IOLegacy` pour avoir les
  fonctionnalités équivalentes.

### Solution 2 (2022+)

Pour gérer tous ces problèmes, Arccon propose maintenant le mécanisme
suivant. Pour chaque package `<pkg>`, on définit les variables CMAKE
suivantes qui sont placées dans le cache. Elles sont donc accessibles
partout:

- `<pkg>_FOUND` : TRUE si le package est trouvé.

- `ARCCON_PACKAGE_<pkg>_TARGETS`: Liste des cibles nécessaires pour
  utiliser le package `<pkg>`. Il est donc possible d'ajouter ces
  cibles en faisant un `target_link_libraries(my_target PUBLIC ${ARCCON_PACKAGE_<pkg>_TARGETS}`).

Il existe trois commandes pour enregistre un package, qui sont
définies dans le fichier `RegisterPackageLibrary.cmake`. Elles
positionnent toutes les 2 variables ci-dessous:

1. `arccon_register_package_library(package_name var_name)`
2. `arccon_register_cmake_config_target(package_name CONFIG_TARGET_NAME target_name)`
3. `arccon_register_cmake_multiple_config_target(package_name CONFIG_TARGET_NAMES target_name1 [target_name2] ...)`

La commande (1) est la commande historique et crée une cible interface
`arcconpkg_<pkg>` à partir des valeurs des variables
`<var_name>_INCLUDE_DIRS` et `<var_name>_LIBARIES`. Un alias **global**
`arccon::<pkg>` est aussi crée.

La commande (2) créé un alias **local** `arccon::<pkg>` au nom de
`<target_name>`.
