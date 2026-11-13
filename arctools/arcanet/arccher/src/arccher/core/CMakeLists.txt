#set(ARCCHER_SOURCES
#  Arct.h
#  Exception.h
#)
#
#add_library(arccher_core ${ARCCHER_SOURCES})
#
#target_include_directories(arccher_core PRIVATE ./../../)
#target_link_libraries(arccher_core PRIVATE
#  Arccore::arccore_trace
#  Arccore::arccore_common
#  Arccore::arccore_base
#)
#
#install(TARGETS arccher_core
#  EXPORT ArcaNetTargets
#  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#)
