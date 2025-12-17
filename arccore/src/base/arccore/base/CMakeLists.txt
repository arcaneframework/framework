if(ARCCORE_ENABLE_GLIB)
  find_package(Glib 2.32 REQUIRED)
endif()

include(srcs.cmake)

arccore_add_component_library(base
  FILES ${SOURCES}
)

target_link_libraries(arccore_base PUBLIC $<BUILD_INTERFACE:arccore_build_compile_flags>)

if(ARCCORE_ENABLE_GLIB)
  target_link_libraries(arccore_base PRIVATE arcconpkg_Glib)
  target_compile_definitions(arccore_base PRIVATE ARCCORE_HAS_GLIB)
endif()

target_compile_features(arccore_base PUBLIC cxx_std_${ARCCORE_CXX_STANDARD})

# Nécessaire pour pouvoir inclure 'arccore_config.h'
target_include_directories(arccore_base PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
