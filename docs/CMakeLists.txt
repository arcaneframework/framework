find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED breathe)

# doxygen settings can be set here, prefixed with "DOXYGEN_"
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# For sphinx
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
set(SPHINX_COPYRIGHT "CEA, IFPEN")
set(SPHINX_AUTHOR "CEA, IFPEN")

# this target will only be built if specifically asked to.
# run "make api-docs" to create the doxygen documentation
doxygen_add_docs(alien_api_docs_doxygen
        ${PROJECT_SOURCE_DIR}/modules
        COMMENT "Generate API-documents for Alien."
        )

sphinx_add_docs(
        alien_api_docs_sphinx
        BREATHE_PROJECTS alien_api_docs_doxygen
        BUILDER html
        SOURCE_DIRECTORY sphinx
        OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/api-docs"
)
