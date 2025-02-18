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

set(SOURCES
  MessagePassingMpiGlobal.h
  MessagePassingMpiGlobal.cc
  MessagePassingMpiEnum.h
  MessagePassingMpiEnum.cc
  MpiAdapter.h
  MpiAdapter.cc
  MpiLock.h
  MpiDatatype.h
  MpiDatatype.cc
  MpiControlDispatcher.cc
  MpiRequestList.cc
  MpiSerializeMessageList.cc
  MpiSerializeDispatcher.cc
  MpiTypeDispatcher.cc
  MpiMessagePassingMng.cc
  MpiMessagePassingMng.h

  StandaloneMpiMessagePassingMng.cc
  StandaloneMpiMessagePassingMng.h

  internal/IMpiProfiling.h
  internal/MpiControlDispatcher.h
  internal/MpiRequest.h
  internal/MpiRequestList.h
  internal/MpiSerializeMessageList.h
  internal/MpiSerializeDispatcher.h
  internal/MpiTypeDispatcher.h
  internal/MpiTypeDispatcherImpl.h
  internal/NoMpiProfiling.h
)

arccore_add_component_library(message_passing_mpi
  FILES ${SOURCES})

target_link_libraries(arccore_message_passing_mpi PUBLIC arccore_trace arccore_message_passing arccore_collections arccore_concurrency)
target_link_libraries(arccore_message_passing_mpi PUBLIC arcconpkg_MPI)
if (ARCCORE_ADDITIONAL_MPI_LIBRARIES)
  target_link_libraries(arccore_message_passing_mpi PUBLIC ${ARCCORE_ADDITIONAL_MPI_LIBRARIES})
endif()
