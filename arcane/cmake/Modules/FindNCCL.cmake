#
# Find the NCCL (NVIDIA Collective Communications Library)

arccon_return_if_package_found(NCCL)

pkg_check_modules(NCCL nccl)
if (NCCL_FOUND)
  message(STATUS "NCCL found via pkg-config")
  # Arccon utilise '*_LIBRARIES" mais ces dernières ne
  # contiennent pas le chemin . On prend 'NCCL_LINK_LIBRARIES' à la place
  set(NCCL_LIBRARIES "${NCCL_LINK_LIBRARIES}")
else()
  message(STATUS "NCCL not found via pkg-config. Trying manual search.")
endif ()

# Il est possible que NCCL ne soit pas installé via pkg-config
# Dans ce cas, on cherche directement les fichiers d'en-tête
# et la bibliothèque associée
if (NOT NCCL_FOUND)
  find_path(NCCL_INCLUDE_DIR nccl.h)
  find_library(NCCL_LIBRARY nccl)
  if (NCCC_INCLUDE_DIR AND NCCL_LIBRARY)
    set(NCCL_LIBRARIES "${NCCL_LIBRARY}")
    set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")
    set(NCCL_FOUND TRUE)
  endif()
endif()

message(STATUS "NCCL_INCLUDE_DIRS   = ${NCCL_INCLUDE_DIRS}")
message(STATUS "NCCL_LIBRARIES      = ${NCCL_LIBRARIES}")
message(STATUS "NCCL_LDFLAGS        = ${NCCL_LDFLAGS}")
message(STATUS "NCCL_LIBDIR         = ${NCCL_LIBDIR}")
message(STATUS "NCCL_PREFIX         = ${NCCL_PREFIX}")
message(STATUS "NCCL_VERSION        = ${NCCL_VERSION}")
message(STATUS "NCCL_FOUND          = ${NCCL_FOUND}")
message(STATUS "NCCL_LINK_LIBRARIES = ${NCCL_LINK_LIBRARIES}")

if (NCCL_FOUND)
  arccon_register_package_library(NCCL NCCL)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
