# ADoc

Un Sample de documentation est disponible dans le dossier d'installation de Arcane (`${ARCANE_PREFIX_DIR}/share/adoc/doc`).
Il suffit de le copier dans les sources du code (par exemple : `${CMAKE_SOURCE_DIR}/doc`) et de modifier
le CMakeLists.txt du projet.

À ajouter dans le CMakeLists.txt :
```cmake
find_package(Doxygen)

# Si le package Doxygen est introuvable, inutile de continuer.
if(Doxygen_FOUND)

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
    
    # Initialisation des variables Doxygen.
    adoc_initialize_vars(${doc_type})
    # Configuration de la génération des infos AXL (facultatif).
    adoc_initialize_axldoc(${doc_type} ${AXLINFO_BINARY} "")
    
    # Si nécessaire, il est possible d'ajouter/modifier/écraser des variables Doxygen pour
    # personnaliser la documentation.
    # Dans ces fichiers, il est aussi possible de définir une variable contenant la
    # liste des dossiers dans lesquelles se trouve les fichiers à include dans la
    # documentation (dans cet exemple, on définit la
    # variable `SAMPLE_DOXYGEN_INPUT`).
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

endif()
```

Deux variables CMake sont disponibles :
- `ADOC_LEGACY_THEME` (`ON`/`OFF`) : permet de passer au thème Doxygen classique (par défaut : `OFF`),
- `ADOC_MATHJAX` (`ON`/`OFF`) : permet de savoir si la doc aura accès à internet ou non (par défaut : `ON`).

Deux cibles seront générées (il sera nécessaire de faire `ninja userdoc` et/ou `ninja devdoc` après la configuration
et la compilation du code) :
- "userdoc" : destinée aux utilisateurs du code. Sa génération est plus rapide que la "dev", elle est plus légère
  étant donné qu'elle n'inclut pas de graphes UML.
- "devdoc" : destinée aux développeurs du code. Cette documentation inclut les éléments privés des classes, en autres.
