if(WIN32)
  return()
endif()
set(PKGS MPI IceT OSMesa OpenGL)
arcane_find_package(MPI)
arcane_find_package(IceT)
arcane_find_package(OSMesa)
if(NOT MPI_FOUND OR NOT TARGET IceT_FOUND OR NOT OSMesa_FOUND)
  return()
endif()
arcane_find_package(OpenGL)
if (NOT TARGET OpenGL::GL OR NOT TARGET OpenGL::GLU)
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_hyoda
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/hyoda
  FILES ${ARCANE_SOURCES}
)

target_link_libraries(arcane_hyoda PUBLIC arcane_core arcane_utils)
target_link_libraries(arcane_hyoda PRIVATE OpenGL::GLU OpenGL::GL)
arcane_add_arccon_packages(arcane_hyoda PRIVATE OSMesa IceT MPI)

arcane_register_library(arcane_hyoda OPTIONAL)

add_subdirectory(gui)
