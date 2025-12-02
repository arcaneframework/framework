#
# Find the NCCL (NVIDIA Collective Communications Library)

arccon_return_if_package_found(NCCL)

pkg_check_modules(NCCL nccl)

message(STATUS "NCCL_INCLUDE_DIRS = ${NCCL_INCLUDE_DIRS}")
message(STATUS "NCCL_LIBRARIES    = ${NCCL_LIBRARIES}")
message(STATUS "NCCL_LDFLAGS      = ${NCCL_LDFLAGS}")
message(STATUS "NCCL_LIBDIR       = ${NCCL_LIBDIR}")
message(STATUS "NCCL_PREFIX       = ${NCCL_PREFIX}")
message(STATUS "NCCL_VERSION      = ${NCCL_VERSION}")
message(STATUS "NCCL_LINK_LIBRARIES = ${NCCL_LINK_LIBRARIES}")

if (NCCL_FOUND)
  # Arccon utilise '*_LIBRARIES" mais ces dernières ne
  # contiennent pas le chemin . On prend 'NCCL_LINK_LIBRARIES' à la place
  set(NCCL_LIBRARIES "${NCCL_LINK_LIBRARIES}")
  arccon_register_package_library(NCCL NCCL)
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
