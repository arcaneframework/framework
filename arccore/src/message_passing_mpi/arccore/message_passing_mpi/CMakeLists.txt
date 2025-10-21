find_package(MPI)

if(NOT MPI_FOUND)
  message(FATAL_ERROR "'MPI' is not available."
    " Add -DARCCORE_USE_MPI=NO to cmake configuration to compile without MPI."
  )
endif()

# On a besoin du support de MPI 3.1
if (NOT WIN32)
  if (MPI_CXX_VERSION)
    if (MPI_CXX_VERSION VERSION_LESS "3.1")
      message(FATAL_ERROR "MPI Version (${MPI_CXX_VERSION}) is too old. Version 3.1 is required."
        " Add -DARCCORE_USE_MPI=NO to cmake configuration to compile without MPI."
      )
    endif()
  endif()
endif()

include(srcs.cmake)

arccore_add_component_library(message_passing_mpi
  FILES ${SOURCES})

target_link_libraries(arccore_message_passing_mpi PUBLIC arccore_trace arccore_message_passing arccore_collections arccore_concurrency)
target_link_libraries(arccore_message_passing_mpi PUBLIC arcconpkg_MPI)
if (ARCCORE_ADDITIONAL_MPI_LIBRARIES)
  target_link_libraries(arccore_message_passing_mpi PUBLIC ${ARCCORE_ADDITIONAL_MPI_LIBRARIES})
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
