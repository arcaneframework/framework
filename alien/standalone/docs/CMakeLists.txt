if (ALIEN_GENERATE_DOCUMENTATION)
  find_package(Doxygen)

  # Si le package Doxygen est introuvable, inutile de continuer.
  if (Doxygen_FOUND)
    block()
      # Variable utile pour les fichiers du sample `CommonDocConfig.cmake`,
      # `UserDocConfig.cmake` et `DevDocConfig.cmake`.
      set(DOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

      # Emplacement des fichiers de configuration.
      set(ADOC_DOC_CONFIG_DIR "${DOC_DIR}")

      function(doc_generation doc_type)
        set(ADOC_DOC_TYPE "${doc_type}")
        set(ADOC_DOC_TARGET "alien${doc_type}doc")
        adoc_generate_doc()
      endfunction()

      # On demande la génération des deux cibles documentations (`alienuserdoc` et `aliendevdoc`).
      doc_generation("user")
      doc_generation("dev")

      # On ajoute une cible pour générer les deux documentations.
      #      add_custom_target(aliendoc DEPENDS alienuserdoc aliendevdoc)
    endblock()
  endif ()
endif (ALIEN_GENERATE_DOCUMENTATION)

#-----------------------------------------------------------
# Tutorial
#-----------------------------------------------------------
if (ALIEN_GENERATE_TUTORIAL)
  add_subdirectory(tutorial)
  install(DIRECTORY tutorial DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif (ALIEN_GENERATE_TUTORIAL)
