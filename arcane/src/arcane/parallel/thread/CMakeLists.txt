set(PKGS Glib TBB)

include(srcs.cmake)

if (ARCCORE_ENABLE_TBB)
  message(STATUS "[arcane_thread] Using OneTBB")
  arcane_find_package(TBB)
  set(ARCANE_TBB_SOURCES TBBThreadImplementation.cc)
  set(ARCANE_HAS_TBBIMPL TRUE)
  set(ARCANE_HAS_ONETBB TRUE)
  list(APPEND ARCANE_SOURCES ${ARCANE_TBB_SOURCES})
  set(ARCANE_HAS_TASKS TRUE CACHE STRING "Support for tasks" FORCE)
endif()

arcane_add_library(arcane_thread
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/thread
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_thread PRIVATE ARCANE_COMPONENT_arcane_thread)
if (ARCANE_HAS_ONETBB)
  target_compile_definitions(arcane_thread PRIVATE ARCANE_USE_ONETBB)
endif()

arcane_add_arccon_packages(arcane_thread PRIVATE ${PKGS})

target_link_libraries(arcane_thread PUBLIC arcane_impl arcane_core arcane_utils)

arcane_register_library(arcane_thread)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
