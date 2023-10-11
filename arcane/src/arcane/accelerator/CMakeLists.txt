# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour les accélérateurs
#
# Pour l'instant seul CUDA est supporté
# Ce fichier doit être inclus après 'arcane/accelerator/cuda' ou
# 'arcane/accelerator/hip'

include(srcs.cmake)

arcane_add_library(arcane_accelerator
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator
  FILES ${ARCANE_SOURCES}
  )

arcane_accelerator_add_source_files(MemoryCopier.cc Scan.cc CommonUtils.cc)

target_link_libraries(arcane_accelerator PUBLIC
  arcane_core
  arcane_accelerator_core
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_cuda>"
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_hip>"
  )

# Cette cible est optionnelle pour éviter que le code n'utilisant
# pas les accélérateurs ait besoin d'un runtime accélérateur
# (par exemple 'cudart')
arcane_register_library(arcane_accelerator OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
