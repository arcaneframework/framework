find_package(MPI)
if(NOT MPI_FOUND)
  message(STATUS "Disabling 'MessagePassingMpi' component because 'MPI' is not available")
  return()
endif()
# On a besoin du support de MPI 3.1
if (MPI_C_VERSION VERSION_LESS "3.1")
  message(STATUS "Disabling 'MessagePassingMpi' component because MPI Version (${MPI_VERSION}) is too old. Version 3.1 is required")
  return()
endif()

set(SOURCES
  MessagePassingMpiGlobal.h
  MessagePassingMpiGlobal.cc
  MpiAdapter.h
  MpiAdapter.cc
  MpiLock.h
  MpiDatatype.h
  MpiDatatype.cc
  MpiTypeDispatcher.h
  MpiTypeDispatcherImpl.h
  MpiTypeDispatcher.cc
  MpiMessagePassingMng.cc
  MpiMessagePassingMng.h
  StandaloneMpiMessagePassingMng.cc
  StandaloneMpiMessagePassingMng.h
)

arccore_add_component_library(message_passing_mpi
  FILES ${SOURCES})

target_link_libraries(arccore_message_passing_mpi PUBLIC arccore_trace arccore_message_passing arccore_collections arccore_concurrency)
target_link_libraries(arccore_message_passing_mpi PUBLIC arcconpkg_MPI)
