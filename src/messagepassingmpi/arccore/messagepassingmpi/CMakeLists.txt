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
)

arccore_add_library(arccore_messagepassingmpi
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/messagepassingmpi
  RELATIVE_PATH arccore/messagepassingmpi
  FILES ${SOURCES}
)

target_compile_definitions(arccore_messagepassingmpi PRIVATE ARCCORE_COMPONENT_arccore_messagepassingmpi)
target_include_directories(arccore_messagepassingmpi PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/messagepassingmpi> $<INSTALL_INTERFACE:include>)
install(TARGETS arccore_messagepassingmpi EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)

target_link_libraries(arccore_messagepassingmpi PUBLIC arccore_trace arccore_messagepassing arccore_collections arccore_concurrency)
target_link_libraries(arccore_messagepassingmpi PUBLIC arcconpkg_MPI)
