add_executable(arcane_driver ArcaneDriverMain.cc)

# TODO: regarder à quoi sert 'DRIVER_LIBS'
target_link_libraries(arcane_driver PUBLIC arcane_driverlib arcane_full ${DRIVER_LIBS})

# Pour mettre l'executable dans 'lib'
arcane_target_set_standard_path(arcane_driver)

set(_LIB_PATH ${CMAKE_BINARY_DIR}/lib)
set_target_properties(arcane_driver PROPERTIES INSTALL_RPATH_USE_LINK_PATH 1)
set_target_properties(arcane_driver PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

configure_file(ArcaneDriver.config ${_LIB_PATH} COPYONLY)

install(TARGETS arcane_driver DESTINATION lib)
install(FILES ArcaneDriver.config DESTINATION lib)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
