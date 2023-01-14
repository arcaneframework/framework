arcane_find_package(HDF5)
if (NOT HDF5_FOUND)
  message(STATUS "Disabling '.Net' wrapping of HDF5 because HDF5 is not available")
  return()
endif()

# ----------------------------------------------------------------------------
# Normalement on doit toujours avoir HDF5_VERSION sauf si on a détecté à la
# main (dans arccon) HDF5. Si la version est connue, alors on vérifie qu'on
# a au moins la 1.10. Si elle n'est pas connue, on affiche un avertissement
# mais on essaie quand même d'utiliser HDF5.
message(STATUS "HDF5_VERSION=${HDF5_VERSION}")
if (HDF5_VERSION)
  if (HDF5_VERSION VERSION_LESS "1.10")
    message(STATUS "Disabling '.Net' wrapping of HDF5 because HDF5 version (${HDF5_VERSION}) is too old (1.10+ required).")
    return()
  endif()
else()
  message(WARNING
    "[wrapper-hdf5] version of HDF5 is unknown. "
    "Version 1.10+ is required. Notwithstanding trying to use HDF5 '.Net' wrapper")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Wrapping '.Net' du support HDF5.

set(ARCANE_SWIG_HDF5_FILES
  ArcaneSwigHdf5.i
  Hdf5Utils.i
)

# ----------------------------------------------------------------------------
# Installe les .cs et les .i (temporaire) pour ceux qui souhaitent wrapper Arcane.
# TODO: A terme, fabriquer un .i pour export.
install(FILES
  ${ARCANE_SWIG_HDF5_FILES}
  DESTINATION share/wrapper/swig/hdf5
  )

arcane_wrapper_add_swig_target(NAME hdf5 SOURCE ArcaneSwigHdf5.i
  NAMESPACE_NAME Arcane.Hdf5
  DLL_NAME Arcane.Hdf5
  SWIG_TARGET_DEPENDS core
  )
target_link_libraries(arcane_dotnet_wrapper_hdf5 PUBLIC arcane_hdf5)
arcane_add_arccon_packages(arcane_dotnet_wrapper_hdf5 PRIVATE HDF5)
# Indique les .i dont dépend 'ArcaneSwigServices.i'
set_property(TARGET arcane_dotnet_wrapper_hdf5 PROPERTY SWIG_DEPENDS ${ARCANE_SWIG_HDF5_FILES})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
