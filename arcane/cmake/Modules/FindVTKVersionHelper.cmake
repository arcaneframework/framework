# Attention, on ne fait pas de find VTK "global" dans Arcane.
# Lorsque l'on souhaite un composent ex. IOXML, IOLegacy, IOXdmf2, il faut le chercher directement.
# En revanche en fonction des versions de vtk (en particulier entre la 8 et la 9) les noms de package changent :
# - vtkIOXML, vtkLegacy, vtkIOXdmf2 dans les versions 8
# - IOXML, IOLegacy, IOXdmf2 dans les versions 9+

# Ce finder est là pour trouver le bon nom de package et le fournir à l'utilisateur
# Le finder va créer la variable VTK_PACKAGE_PREFIX. Il faut ensuite l'utiliser pour trouver le composant souhaité :
# arcane_find_package(VTK)
# arcane_find_package(${VTK_PACKAGE_PREFIX}IOXML)

if (VTK_PACKAGE_PREFIX)
  return()
endif ()

find_package(VTK QUIET)

if (NOT VTK_FOUND)
  message(STATUS "VTK not found")
  return()
endif ()

# Handle VTK version (if > 8.90 targets are VTK::Component, if not vtkComponent
if (VTK_VERSION VERSION_GREATER "8.90.0")
  set(VTK_PACKAGE_PREFIX "")
  set(VTK_PACKAGE_PREFIX "" PARENT_SCOPE)
else ()
  set(VTK_PACKAGE_PREFIX "vtk")
  set(VTK_PACKAGE_PREFIX "vtk" PARENT_SCOPE)
endif ()

set(VTKVersionHelper_FOUND TRUE)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
