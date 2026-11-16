set(ARCCHER_SOURCES
  Reproc.cc
  Reproc.h
)

set(REPROC++ ON)

add_subdirectory(reproc)

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
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
