find_package(Glib REQUIRED)

set(SOURCES
  TraceGlobal.h
  TraceGlobal.cc
  Trace.h
  TraceClassConfig.h
  TraceMessage.h
  TraceMessage.cc
  ITraceMng.h
)

arccore_add_library(arccore_trace
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/trace
  RELATIVE_PATH arccore/trace
  FILES ${SOURCES}
)

target_link_libraries(arccore_trace PRIVATE arcconpkg_Glib)
target_compile_definitions(arccore_trace PRIVATE ARCCORE_COMPONENT_arccore_trace)
target_include_directories(arccore_trace PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/trace> $<INSTALL_INTERFACE:include>)
install(TARGETS arccore_trace EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)

target_link_libraries(arccore_trace PUBLIC arccore_base)
