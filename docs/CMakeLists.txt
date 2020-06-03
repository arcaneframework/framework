find_package(Doxygen REQUIRED)

# doxygen settings can be set here, prefixed with "DOXYGEN_"
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/api-docs")

# this target will only be built if specifically asked to.
# run "make api-docs" to create the doxygen documentation
doxygen_add_docs(alien_api_docs
        ${PROJECT_SOURCE_DIR}/modules
        COMMENT "Generate API-documents for Alien."
        )