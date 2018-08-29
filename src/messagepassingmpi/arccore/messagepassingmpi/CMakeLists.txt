find_package(MPI)
if(NOT MPI_FOUND)
  message(STATUS "Disabling 'MessagePassingMpi' component because 'MPI' is not available")
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

arccore_add_component_library(messagepassingmpi
  FILES ${SOURCES})

target_link_libraries(arccore_messagepassingmpi PUBLIC arccore_trace arccore_messagepassing arccore_collections arccore_concurrency)
target_link_libraries(arccore_messagepassingmpi PUBLIC arcconpkg_MPI)
