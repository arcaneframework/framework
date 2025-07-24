# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane,IOXdmf2 est utilisé pour la gestion des Xdmf
# Voir fichier 'FindIOXML' pour plus d'infos
# Ce finder et utilisé pour des versions de vtk >= 9. Auparavant le package s'appelait vtkIOXdmf2
arccon_return_if_package_found(IOXdmf2)

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS IOXdmf2)

if (VTK_FOUND)
  message(STATUS "VTK (for IOXdmf2) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

if(TARGET VTK::IOXdmf2)
  set(IOXdmf2_FOUND TRUE)
  get_target_property(IOXdmf2_INCLUDE_DIRS VTK::IOXdmf2 INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "IOXdmf2_INCLUDE_DIRS = ${IOXdmf2_INCLUDE_DIRS}")
  arcane_vtkutils_add_depend_lib_to_list_v2(VTK::IOXdmf2)
  message(STATUS "IOXML LIBS=${_ALLLIBS}")
  set(IOXdmf2_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(IOXdmf2 IOXdmf2)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::IOXdmf2 ALIAS arcconpkg_IOXdmf2)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
