# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour les accélérateurs
#
# Pour l'instant seul CUDA est supporté
# Ce fichier doit être inclue après 'arcane/accelerator/cuda'

include(srcs.cmake)

arcane_add_library(arcane_accelerator
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator
  FILES ${ARCANE_SOURCES}
  )

target_link_libraries(arcane_accelerator PUBLIC
  arcane_core
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_cuda>"
  )

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
