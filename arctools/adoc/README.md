# ADoc (Arcane Documentation)

## Introduction

Ce projet vise à simplifier l'utilisation du thème Doxygen utilisé par la documentation Arcane pour les projets
utilisant Arcane.

Au lieu d'utiliser un doxyfile déjà rempli, on va demander à CMake d'en générer un.
CMake utilisera les variables commençant par `DOXYGEN_` pour compléter ce fichier.

ADoc fourni plusieurs variables `DOXYGEN_` permettant à Doxygen de trouver le thème.
Il se charge aussi de générer les pages contenant les informations sur les modules et services à l'aide de AXLDoc.

Trois fonctions CMake sont disponibles :

- `adoc_initialize(doc_type)` :
  Fonction permettant de définir les variables DOXYGEN pour générer la
  documentation ADoc et de générer les fichiers nécessaires.

  Deux valeurs sont possibles pour "doc_type" : "user" et "dev".

  La documentation "user" est destinée aux utilisateurs du code. Sa génération
  est plus rapide que la "dev", elle est plus légère étant donné qu'elle
  n'inclut pas de graphes UML.

  La documentation "dev" est destinée aux développeurs du code.
  Cette documentation inclut les éléments privés des classes, en autres.

  Voir les fichiers "ADocCommonVars.cmake", "ADocUserVars.cmake" et
  "ADocDevVars.cmake" pour plus de détails.

- `adoc_initialize_axldoc(doc_type, executable config_file_dir)` :
  Fonction permettant de configurer le script permettant de générer les
  informations sur les AXL.

  Deux valeurs sont possibles pour "doc_type" : "user" et "dev". Cela permet
  de savoir quelles informations récupérer des fichiers AXL.

  "executable" correspond au chemin de l'exécutable généré par le code. Il
  sera lancé avec un mode spécial de Arcane permettant de générer les
  informations AXL des modules et services.

  "config_file_dir" correspond au répertoire où est situé le fichier de
  configuration de l'exécutable (`File.config`). Il est possible de fournir une
  chaine de caractère vide pour choisir le répertoire de l'exécutable.

  Sous Windows, cette fonction n'a aucun effet.

- `adoc_link_axldoc_doxygen(doc_type, doxygen_target_name)` :
  Fonction permettant d'ajouter la génération des informations AXL comme
  dépendance à la génération de la documentation.

  Deux valeurs sont possibles pour "doc_type" : "user" et "dev".

  "doxygen_target_name" correspond au nom de la cible utilisé par la
  commande "doxygen_add_docs()".

  Sous Windows, cette fonction n'a aucun effet.

---

## Utilisation

Un Sample de documentation est disponible dans le dossier d'installation de Arcane (
`${ARCANE_PREFIX_DIR}/share/adoc/doc`).
Il suffit de le copier dans les sources du code (par exemple : `${CMAKE_SOURCE_DIR}/doc`) et de modifier
le CMakeLists.txt du projet.

À ajouter dans le CMakeLists.txt :

```cmake
# Répertoire où seront regroupés tous les fichiers .axl.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/axl)

find_package(Doxygen)

# Si le package Doxygen est introuvable, inutile de continuer.
if (Doxygen_FOUND)

  # Emplacement du dossier "doc" copié précédemment. TODO À personnaliser.
  set(DOC_DIR "${CMAKE_SOURCE_DIR}/doc")

  # Choisir l'exécutable généré (facultatif). TODO À personnaliser.
  # Arcane l'utilisera pour générer les informations sur les services et les
  # modules.
  set(AXLINFO_BINARY "${CMAKE_BINARY_DIR}/bin/Nonreg")

  # Ce fichier contient les fonctions cmake ADoc.
  include(${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocConfig.cmake)

  # On utilise une fonction pour réduire le scope des variables DOXYGEN_ et pour
  # générer les documentations "user" et "dev".
  function(adoc_generation doc_type)

    # Pour générer la documentation, on utilisera une cible du nom de "userdoc"/"devdoc".
    set(DOC_TARGET "${doc_type}doc")

    # Initialisation des variables Doxygen. C'est ici que l'on définit, entre autres choses,
    # l'emplacement de tous les fichiers du thème utilisé par Arcane. 
    adoc_initialize(${doc_type})
    # Configuration de la génération des infos AXL (facultatif).
    adoc_initialize_axldoc(${doc_type} ${AXLINFO_BINARY} "")

    # Si nécessaire, il est possible d'ajouter/modifier/écraser des variables Doxygen pour
    # personnaliser la documentation.
    # Les variables défines par ADoc sont situées dans les fichiers :
    # - ${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocCommonVars.cmake
    # - ${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocUserVars.cmake
    # - ${ARCANE_PREFIX_DIR}/share/adoc/cmake/ADocDevVars.cmake
    #
    # Les fichiers "SampleCommonDocConfig.cmake", "SampleUserDocConfig.cmake" et
    # "SampleDevDocConfig.cmake" servent justement à ça !
    # On en profite aussi pour définir une variable contenant la
    # liste des dossiers dans lesquelles se trouvent les fichiers à include dans la
    # documentation (dans cet exemple, on définit la
    # variable `SAMPLE_DOXYGEN_INPUT`).
    #
    # Pour un exemple plus complet, il est possible d'aller voir les fichiers :
    # - "arcane/doc/ArcaneCommonDocConfig.cmake"
    # - "arcane/doc/ArcaneUserDocConfig.cmake"
    # - "arcane/doc/ArcaneDevDocConfig.cmake"
    include(${DOC_DIR}/SampleCommonDocConfig.cmake)
    if (${doc_type} STREQUAL "user")
      include(${DOC_DIR}/SampleUserDocConfig.cmake)
    else ()
      include(${DOC_DIR}/SampleDevDocConfig.cmake)
    endif ()

    # On définit le dossier de sortie pour la documentation.
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/share/${DOC_TARGET}")

    # Pour afficher le lien vers la documentation générée, on doit utiliser une
    # cible intermédiaire.
    set(INTERNAL_DOC_TARGET "_${DOC_TARGET}")
    doxygen_add_docs(
      ${INTERNAL_DOC_TARGET}
      ${SAMPLE_DOXYGEN_INPUT}
    )
    # On ajoute une dépendance pour ${INTERNAL_DOC_TARGET} qui est la génération
    # des infos AXL (facultatif).
    adoc_link_axldoc_doxygen(${doc_type} ${INTERNAL_DOC_TARGET})

    # On donne, dans les logs, l'emplacement de l'index.html généré.
    add_custom_target(${DOC_TARGET} COMMAND echo "Doc index file : file://${DOXYGEN_OUTPUT_DIRECTORY}/html/index.html")
    add_dependencies(${DOC_TARGET} ${INTERNAL_DOC_TARGET})
  endfunction()

  # On demande la génération des deux cibles documentations (`userdoc` et `devdoc`).
  adoc_generation("user")
  adoc_generation("dev")

endif ()
```

Des variables CMake sont disponibles :

- `ADOC_LEGACY_THEME` (`ON`/`OFF`) : permet de passer au thème Doxygen classique (par défaut : `OFF`),
- `ADOC_MATHJAX` (`ON`/`OFF`) : permet d'activer MathJax (par défaut : `ON`),
- `ADOC_PROJECT_REPO_LINK` (url) : permet de définir le lien vers le dépot du code,
- `ADOC_PROJECT_ICON` (chemin de l'icône) : permet de définir une icône pour la page web (équivalent à l'option Doxygen
  `PROJECT_ICON` mais qui fonctionne) (trois formats supportés : `svg`, `png` et `webp`).

Deux cibles seront générées (il sera nécessaire de faire `ninja userdoc` et/ou `ninja devdoc` après la configuration
et la compilation du code) :

- "userdoc" : destinée aux utilisateurs du code. Sa génération est plus rapide que la "dev", elle est plus légère
  étant donné qu'elle n'inclut pas de graphes UML.
- "devdoc" : destinée aux développeurs du code. Cette documentation inclut les éléments privés des classes, en autres.
