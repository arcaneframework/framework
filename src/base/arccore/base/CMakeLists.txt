find_package(Glib REQUIRED)

set(SOURCES
  ArccoreGlobal.h
  ArccoreGlobal.cc
  ArrayIterator.h
  ArrayRange.h
  ArrayView.h
  ArrayView.cc
  BaseTypes.h
  BasicTranscoder.h
  BasicTranscoder.cc
  BaseTypes.h
  CoreArray.h
  Iterator.h
)

message(STATUS "BinaryDir= ${Arccore_BINARY_DIR}")

arccore_add_library(arccore_base
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/base
  RELATIVE_PATH arccore/base
  FILES ${SOURCES}
)

target_link_libraries(arccore_base PRIVATE arccon::Glib)
target_compile_definitions(arccore_base PRIVATE ARCCORE_COMPONENT_arccore_base)
target_include_directories(arccore_base PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/base> $<INSTALL_INTERFACE:include>)
target_include_directories(arccore_base PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
install(TARGETS arccore_base EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)
