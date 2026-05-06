cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(generated_config_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Pour Arcane.
set(adoc_source_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(axldoc_path ${CMAKE_BINARY_DIR}/bin/axldoc)
configure_file(cmake/ADocConfig.cmake.in ${CMAKE_BINARY_DIR}/share/adoc/cmake/ADocConfig.cmake @ONLY)

# Pour les projets basés sur Arcane.
set(adoc_source_dir ${CMAKE_INSTALL_PREFIX}/share/adoc)
set(axldoc_path ${CMAKE_INSTALL_PREFIX}/bin/axldoc)
#configure_file(cmake/ADocConfig.cmake.in ${generated_config_dir}/ADocConfig.cmake @ONLY)

install(DIRECTORY theme/ DESTINATION share/adoc/theme)
install(DIRECTORY sample_doc/ DESTINATION share/adoc/sample_doc)
install(FILES generate_axl_infos.sh.in DESTINATION share/adoc)

install(FILES
  cmake/ADocCommonVars.cmake
  cmake/ADocUserVars.cmake
  cmake/ADocDevVars.cmake
  DESTINATION share/adoc/cmake
)

configure_package_config_file(
  "cmake/ADocConfig.cmake.in"
  "${generated_config_dir}/ADocConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ADoc"
)

install(FILES
  ${generated_config_dir}/ADocConfig.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ADoc"
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
