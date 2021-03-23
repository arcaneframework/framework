# Pour compiler Arcane avec les sources CEA
set(ARCANE_CEA_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "Arcane CEA source path" FORCE)
set(ARCANE_ADDITIONAL_PACKAGES SUPERLU CUDA SLOOP F90 LIMA CACHE PATH "Arcane additional packages" FORCE)
set(ARCANE_ADDITIONAL_EXTERNAL_PACKAGES SUPERLU SLOOP F90 LIMA CACHE PATH "Arcane additional external packages" FORCE)
set(ARCANE_ADDITIONAL_COMPONENTS arcane_cea arcane_materials arcane_cea_geometric CACHE STRING "Arcane additional components" FORCE)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
