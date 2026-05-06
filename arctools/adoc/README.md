# ADoc (Arcane Documentation)

## Introduction

Ce projet vise à simplifier l'utilisation du thème Doxygen utilisé par la documentation Arcane pour les projets
utilisant Arcane.

Au lieu d'utiliser un doxyfile déjà rempli, on va demander à CMake d'en générer un.
CMake utilisera les variables commençant par `DOXYGEN_` pour compléter ce fichier.

ADoc fourni plusieurs variables `DOXYGEN_` permettant à Doxygen de trouver le thème.
Il se charge aussi de générer les pages contenant les informations sur les modules et services à l'aide de AXLDoc.

Les fonctions ADoc sont disponibles dans le fichier `ADocConfig.cmake`.

Deux types de documentations sont disponibles :

- "user" : destinée aux utilisateurs du code. Sa génération est plus rapide que la "dev", elle est plus légère
  étant donné qu'elle n'inclut pas de graphes UML.
- "dev" : destinée aux développeurs du code. Cette documentation inclut les éléments privés des classes, en autres.

----

## Variables CMake utilisable

(Aujourd'hui, toutes ces variables sont facultatives)

Ces variables CMake sont disponibles pour personnaliser la génération :

- `ADOC_BUILD_DIR` (par défaut : `${CMAKE_BINARY_DIR}/share/adoc`) : dossier où seront mis les fichiers temporaires
  servant à générer la documentation,
- `ADOC_DOC_TYPE` (`user`/`dev`) (par défaut : `user`) : le type de documentation à générer,
- `ADOC_DOC_TARGET` (par défaut : `userdoc`) : le nom de la cible qui sera générée,
- `ADOC_EXECUTABLE_AXL_GENERATION` (chemin de l'exécutable) (par défaut : vide) : le chemin de l'exécutable qui servira
  à générer les informations des services et des modules (si vide, alors ces informations ne seront pas générées),
- `ADOC_CONFIG_DIR_EXECUTABLE_AXL_GENERATION` (par défaut : chemin du dossier contenant l'exécutable) : chemin du
  dossier contenant le fichier `.config` de l'exécutable,
- `ADOC_LEGACY_THEME` (`ON`/`OFF`) (par défaut : `OFF`) : permet de passer au thème Doxygen classique,
- `ADOC_MATHJAX` (`ON`/`OFF`) (par défaut : `ON`) : permet d'activer MathJax,
- `ADOC_PROJECT_REPO_LINK` (url) (par défaut : vide) : permet de définir le lien vers le dépot du code,
- `ADOC_PROJECT_ICON` (chemin de l'icône) (par défaut : vide) : permet de définir une icône pour la page web (équivalent
  à l'option Doxygen `PROJECT_ICON` mais qui fonctionne) (trois formats supportés : `svg`, `png` et `webp`),

Ces variables seront utilisées uniquement par la fonction `adoc_generate_doc` :

- `ADOC_DOC_CONFIG_DIR` (par défaut : vide) : le dossier contenant les fichiers `CommonDocConfig.cmake`,
  `UserDocConfig.cmake` et `DevDocConfig.cmake`,
- `ADOC_DOXYGEN_INPUT` (par défaut : vide) : permet de spécifier les fichiers/dossiers à inclure dans la documentation
  (équivalent à l'option Doxygen `INPUT`). À noter qu'elle est présente dans les fichiers `UserDocConfig.cmake` et
  `DevDocConfig.cmake` du Sample.

Une variable CMake Doxygen peut être cité ici (utilisée uniquement par la fonction `adoc_generate_doc`) :

- `DOXYGEN_OUTPUT_DIRECTORY` (par défaut : `${CMAKE_BINARY_DIR}/share/${ADOC_DOC_TARGET}`) : le répertoire où sera
  générée la documentation.

----

## Utilisation

Un Sample de documentation est disponible dans le dossier d'installation de Arcane (
`${ARCANE_PREFIX_DIR}/share/adoc/sample_doc`).
Il suffit de copier son contenu dans les sources du code (par exemple dans : `${CMAKE_SOURCE_DIR}/doc/`) et de modifier
le CMakeLists.txt du projet (il est aussi possible de créer un `CMakeLists.txt` dans le dossier
`${CMAKE_SOURCE_DIR}/doc/`).

Il y a deux façons d'utiliser ADoc : en appelant une seule fonction qui se charge de tous les appels ou en appelant les
fonctions de ADoc à la main.

----

### Utilisation simple

À ajouter dans le `CMakeLists.txt` :

```cmake
# Répertoire où seront regroupés tous les fichiers .axl.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/axl)

find_package(Doxygen)

# Si le package Doxygen est introuvable, inutile de continuer.
if (Doxygen_FOUND)
  block()
    # Emplacement du dossier "doc" copié précédemment. TODO À personnaliser.
    # Variable utile pour les fichiers du sample `CommonDocConfig.cmake`,
    # `UserDocConfig.cmake` et `DevDocConfig.cmake`.
    set(DOC_DIR "${CMAKE_SOURCE_DIR}/doc")

    # Emplacement des fichiers de configuration.
    set(ADOC_DOC_CONFIG_DIR "${DOC_DIR}")

    # Choisir l'exécutable généré (facultatif). TODO À personnaliser.
    # Arcane l'utilisera pour générer les informations sur les services et les
    # modules.
    set(ADOC_EXECUTABLE_AXL_GENERATION "${CMAKE_BINARY_DIR}/bin/Nonreg")

    # Ce fichier contient les fonctions cmake ADoc.
    include(${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocConfig.cmake)

    function(doc_generation doc_type)
      set(ADOC_DOC_TYPE "${doc_type}")
      adoc_generate_doc()
    endfunction()

    # On demande la génération des deux cibles documentations (`userdoc` et `devdoc`).
    doc_generation("user")
    doc_generation("dev")
  endblock()
endif ()
```

Une fois le `CMakeLists.txt` modifié, il est important de modifier les fichiers `UserDocConfig.cmake` et
`DevDocConfig.cmake` pour personnaliser les fichiers sources à prendre en compte pour la documentation. Notamment la
variable `ADOC_DOXYGEN_INPUT` qui contient la liste des dossiers contenant les sources à documenter.

Il est aussi possible de personnaliser les variables `DOXYGEN_` en dehors des fichiers `UserDocConfig.cmake` /
`DevDocConfig.cmake` avant l'appel à `adoc_generate_doc()`.

----

### Utilisation avancée

Il est possible d'utiliser les autres fonctions de ADoc à la main, si nécessaire.

À ajouter dans le `CMakeLists.txt` :

```cmake
# Répertoire où seront regroupés tous les fichiers .axl.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/axl)

find_package(Doxygen)

# Si le package Doxygen est introuvable, inutile de continuer.
if (Doxygen_FOUND)

  # Emplacement du dossier "doc" copié précédemment. TODO À personnaliser.
  # Variable utile pour les fichiers du sample `CommonDocConfig.cmake`,
  # `UserDocConfig.cmake` et `DevDocConfig.cmake`.
  set(DOC_DIR "${CMAKE_SOURCE_DIR}/doc")

  # Emplacement des fichiers de configuration.
  set(ADOC_DOC_CONFIG_DIR "${DOC_DIR}")

  # Choisir l'exécutable généré (facultatif). TODO À personnaliser.
  # Arcane l'utilisera pour générer les informations sur les services et les
  # modules.
  set(ADOC_EXECUTABLE_AXL_GENERATION "${CMAKE_BINARY_DIR}/bin/Nonreg")

  # Ce fichier contient les fonctions cmake ADoc.
  include(${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocConfig.cmake)

  # On utilise une fonction pour réduire le scope des variables DOXYGEN_ et pour
  # générer les documentations "user" et "dev".
  function(doc_generation doc_type)

    set(ADOC_DOC_TYPE "${doc_type}")

    # Pour générer la documentation, on utilisera une cible du nom de "userdoc"/"devdoc".
    set(DOC_TARGET "${ADOC_DOC_TYPE}doc")

    # Les fichiers "CommonDocConfig.cmake", "UserDocConfig.cmake" et
    # "DevDocConfig.cmake" servent à personnaliser la documentation.
    # Pour un exemple plus complet, il est possible d'aller voir les fichiers :
    # - "arcane/doc/CommonDocConfig.cmake"
    # - "arcane/doc/UserDocConfig.cmake"
    # - "arcane/doc/DevDocConfig.cmake"
    if (ADOC_DOC_CONFIG_DIR)
      include(${ADOC_DOC_CONFIG_DIR}/CommonDocConfig.cmake)
      adoc_commondoc_config_adoc_variables()
      if (${ADOC_DOC_TYPE} STREQUAL "user")
        include(${ADOC_DOC_CONFIG_DIR}/UserDocConfig.cmake)
        adoc_userdoc_config_adoc_variables()
      else ()
        include(${ADOC_DOC_CONFIG_DIR}/DevDocConfig.cmake)
        adoc_devdoc_config_adoc_variables()
      endif ()
    endif ()

    # Initialisation des variables Doxygen. C'est ici que l'on définit, entre autres choses,
    # l'emplacement de tous les fichiers du thème utilisé par Arcane.
    adoc_initialize()

    if (ADOC_DOC_CONFIG_DIR)
      adoc_commondoc_config_doxygen_variables()
      if (${ADOC_DOC_TYPE} STREQUAL "user")
        adoc_userdoc_config_doxygen_variables()
      else ()
        adoc_devdoc_config_doxygen_variables()
      endif ()
    endif ()

    # On définit le dossier de sortie pour la documentation.
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/share/${DOC_TARGET}")

    # Pour afficher le lien vers la documentation générée, on doit utiliser une
    # cible intermédiaire.
    set(INTERNAL_DOC_TARGET "_${DOC_TARGET}")

    # Ici, la variable est défini dans "UserDocConfig.cmake" et "DevDocConfig.cmake".
    doxygen_add_docs(
      ${INTERNAL_DOC_TARGET}
      ${ADOC_DOXYGEN_INPUT}
    )
    # On ajoute une dépendance pour ${INTERNAL_DOC_TARGET} qui est la génération
    # des infos AXL (facultatif).
    adoc_link_axldoc_doxygen(${INTERNAL_DOC_TARGET})

    # On donne, dans les logs, l'emplacement de l'index.html généré.
    add_custom_target(${DOC_TARGET} COMMAND echo "Doc index file : file://${DOXYGEN_OUTPUT_DIRECTORY}/html/index.html")
    add_dependencies(${DOC_TARGET} ${INTERNAL_DOC_TARGET})
  endfunction()

  # On demande la génération des deux cibles documentations (`userdoc` et `devdoc`).
  doc_generation("user")
  doc_generation("dev")

endif ()
```
