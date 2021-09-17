if (ALIEN_GENERATE_DOCUMENTATION)
    find_package(Doxygen REQUIRED)

    # doxygen settings can be set here, prefixed with "DOXYGEN_"
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

    # For sphinx
    set(DOXYGEN_GENERATE_HTML NO)
    set(DOXYGEN_GENERATE_XML YES)
    find_package(Sphinx REQUIRED breathe)

    set(SPHINX_COPYRIGHT "CEA, IFPEN")
    set(SPHINX_AUTHOR "CEA, IFPEN")

    # this target will only be built if specifically asked to.
    # run "make api-docs" to create the doxygen documentation
    #doxygen_add_docs(alien_api_docs_doxygen
    #        ${PROJECT_SOURCE_DIR}/modules
    #        COMMENT "Generate API-documents for Alien."
    #        )

    sphinx_add_docs(
            alien_doc
            BREATHE_PROJECTS doxygen_api_docs_movesemantic
            BUILDER html
            SOURCE_DIRECTORY sphinx
            OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/alien_doc"
    )
endif (ALIEN_GENERATE_DOCUMENTATION)

#-----------------------------------------------------------
# Tutorial
#-----------------------------------------------------------
if (ALIEN_GENERATE_TUTORIAL)
    add_subdirectory(tutorial)
    install(DIRECTORY tutorial DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif (ALIEN_GENERATE_TUTORIAL)