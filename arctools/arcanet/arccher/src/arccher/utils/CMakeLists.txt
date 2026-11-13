set(ARCCHER_SOURCES
  ArctReader.cc
  ArctReader.h
  CmdLine.h
  CmdLine.cc
)

add_library(arccher_utils ${ARCCHER_SOURCES})

target_include_directories(arccher_utils PRIVATE ./../../)
target_link_libraries(arccher_utils PRIVATE
  arccher_core
  Arccore::arccore_trace
  Arccore::arccore_common
  Arccore::arccore_base
  reproc++
)

install(TARGETS arccher_utils
  EXPORT ArcaNetTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
