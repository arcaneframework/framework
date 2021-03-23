add_executable(arcane_meshutils ${CMAKE_CURRENT_SOURCE_DIR}/ArcaneMeshUtilsMain.cc)
target_link_libraries(arcane_meshutils PUBLIC ${ARCANE_LIBRARIES})

set_target_properties(arcane_meshutils PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ARCANEBUILDROOT}/bin)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ArcaneMeshUtils.config ${CMAKE_CURRENT_BINARY_DIR}/ArcaneMeshUtils.config @ONLY)
