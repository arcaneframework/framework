# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, vtkIOXdmf2 est utilisé pour la gestion des Xdmf
# Voir fichier 'FindvtkIOXML' pour plus d'infos
arccon_return_if_package_found(vtkIOXdmf2)

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS vtkIOXdmf2)

if (VTK_FOUND)
  message(STATUS "VTK (for vtkIOXdmf2) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

if(TARGET vtkIOXdmf2)
  set(vtkIOXdmf2_FOUND TRUE)
  message(STATUS "vtkIOXdmf2_INCLUDE_DIRS = ${vtkIOXdmf2_INCLUDE_DIRS}")
  arcane_vtkutils_add_depend_lib_to_list(vtkIOXdmf2)
  message(STATUS "vtkIOXML LIBS=${_ALLLIBS}")
  set(vtkIOXdmf2_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(vtkIOXdmf2 vtkIOXdmf2)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::vtkIOXdmf2 ALIAS arcconpkg_vtkIOXdmf2)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
