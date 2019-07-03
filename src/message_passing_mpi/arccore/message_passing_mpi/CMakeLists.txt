find_package(MPI)
if(NOT MPI_FOUND)
  message(STATUS "Disabling 'MessagePassingMpi' component because 'MPI' is not available")
  return()
endif()

set(SOURCES
  IMpiProfiling.h
  MessagePassingMpiGlobal.h
  MessagePassingMpiGlobal.cc
  MessagePassingMpiEnum.h
  MessagePassingMpiEnum.cc
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
  NoMpiProfiling.h
  StandaloneMpiMessagePassingMng.cc
  StandaloneMpiMessagePassingMng.h
)

arccore_add_component_library(message_passing_mpi
  FILES ${SOURCES})

target_link_libraries(arccore_message_passing_mpi PUBLIC arccore_trace arccore_message_passing arccore_collections arccore_concurrency)
target_link_libraries(arccore_message_passing_mpi PUBLIC arcconpkg_MPI)
