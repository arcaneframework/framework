find_package(Glib REQUIRED)

set(SOURCES
  ConcurrencyGlobal.h
  ConcurrencyGlobal.cc
  IThreadImplementation.h
  IThreadBarrier.h
  Mutex.h
  Mutex.cc
)

arccore_add_library(arccore_concurrency
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/concurrency
  RELATIVE_PATH arccore/concurrency
  FILES ${SOURCES}
)

target_compile_definitions(arccore_concurrency PRIVATE ARCCORE_COMPONENT_arccore_concurrency)
target_include_directories(arccore_concurrency PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/concurrency> $<INSTALL_INTERFACE:include>)
install(TARGETS arccore_concurrency EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)

target_link_libraries(arccore_concurrency PUBLIC arccore_base)
