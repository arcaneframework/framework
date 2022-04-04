# Find the OTF2 includes and library
include(${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)
arccon_find_legacy_package(NAME Otf2 LIBRARIES otf2 HEADERS otf2/otf2.h)
