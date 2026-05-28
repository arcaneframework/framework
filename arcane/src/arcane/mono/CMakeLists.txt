# WARNING: This component is no longer used
message(FATAL_ERROR "The component 'arcane_mono' is no longer supported")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(PKGS MonoEmbed Glib)
arcane_find_package(MonoEmbed)
if (NOT MonoEmbed_FOUND)
  message(STATUS "Disabling 'arcane_mono' because MonoEmbed is not found")
  return()
endif ()

include(srcs.cmake)

arcane_add_library(arcane_mono
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/mono
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_mono PRIVATE ${PKGS})

target_compile_definitions(arcane_mono PRIVATE ARCANE_COMPONENT_arcane_mono)
# With mono 6.0+, attaching TBB threads causes deadlocks
# in the 'task2_cs' test. These deadlocks do not occur if we do not attach
# the threads. It should be studied whether the attach call is still necessary.
if (MONOEMBEDPKG_VERSION VERSION_GREATER_EQUAL "6.0")
  message(STATUS "[arcane_mono] Do not attach threads")
  target_compile_definitions(arcane_mono PRIVATE ARCANE_MONO_NO_THREAD_ATTACH)
endif ()

target_link_libraries(arcane_mono PRIVATE arcane_utils)

arcane_register_library(arcane_mono OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
