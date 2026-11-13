set(ARCCHER_SOURCES
  Reproc.cc
  Reproc.h
  Run.h
)

set(REPROC++ ON)


add_library(arccher_run ${ARCCHER_SOURCES})

target_include_directories(arccher_run PRIVATE
  ./../../
  reproc/reproc++/include
)

target_link_libraries(arccher_run PRIVATE
  Arccore::arccore_trace
  Arccore::arccore_common
  Arccore::arccore_base
  reproc++
)

install(TARGETS arccher_run
  EXPORT ArcaNetTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
