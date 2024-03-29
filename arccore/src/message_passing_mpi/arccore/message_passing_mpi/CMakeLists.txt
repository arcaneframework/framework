find_package(MPI)

if(NOT MPI_FOUND)
  message(FATAL_ERROR "'MPI' is not available."
    " Add -DARCCORE_USE_MPI=NO to cmake configuration to compile without MPI."
  )
endif()

# On a besoin du support de MPI 3.1
if (NOT WIN32)
  if (MPI_CXX_VERSION VERSION_LESS "3.1")
    message(FATAL_ERROR "MPI Version (${MPI_CXX_VERSION}) is too old. Version 3.1 is required."
      " Add -DARCCORE_USE_MPI=NO to cmake configuration to compile without MPI."
    )
  endif()
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
  MpiControlDispatcher.h
  MpiControlDispatcher.cc
  MpiRequest.h
  MpiRequestList.h
  MpiRequestList.cc
  MpiSerializeMessageList.h
  MpiSerializeMessageList.cc
  MpiSerializeDispatcher.h
  MpiSerializeDispatcher.cc
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
