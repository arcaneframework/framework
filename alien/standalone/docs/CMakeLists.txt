if (ALIEN_GENERATE_DOCUMENTATION)
    find_package(Sphinx REQUIRED breathe exhale)

    set(SPHINX_COPYRIGHT "CEA, IFPEN")
    set(SPHINX_AUTHOR "CEA, IFPEN")

    # this target will only be built if specifically asked to.
    # run "make api-docs" to create the doxygen documentation
    #doxygen_add_docs(alien_api_docs_doxygen
    #        ${PROJECT_SOURCE_DIR}/modules
    #        COMMENT "Generate API-documents for Alien."
    #        )

    #SET(ALIEN_OUTPUT_DIR ${PROJECT_BINARY_DIR}/alien_doc)
    SET(ALIEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/share/aliendoc/html)

    sphinx_add_docs(
            aliendoc
            BREATHE_PROJECTS doxygen_api_docs_movesemantic doxygen_api_docs_refsemantic doxygen_api_docs_core_data
            BUILDER html
            SOURCE_DIRECTORY sphinx
            OUTPUT_DIRECTORY "${ALIEN_OUTPUT_DIR}"
    )
endif (ALIEN_GENERATE_DOCUMENTATION)

#-----------------------------------------------------------
# Tutorial
#-----------------------------------------------------------
if (ALIEN_GENERATE_TUTORIAL)
    add_subdirectory(tutorial)
    install(DIRECTORY tutorial DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif (ALIEN_GENERATE_TUTORIAL)
