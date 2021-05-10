message(STATUS "Checking 'arcane_dotnet_coreclr'")
if(NOT ARCCON_DOTNET_HAS_RUNTIME_coreclr)
  message(STATUS "Disabling 'arcane_dotnet_coreclr' because 'dotnet' exec is not available")
  return()
endif()

set(PKGS CoreClrEmbed)
arcane_find_package(CoreClrEmbed)
if (NOT CoreClrEmbed_FOUND)
  message(STATUS "Disabling 'arcane_dotnet_coreclr' because package 'CoreClrEmbed' is not found")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_dotnet_coreclr
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/dotnet/coreclr
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_dotnet_coreclr PRIVATE ${PKGS})
target_compile_definitions(arcane_dotnet_coreclr PRIVATE ARCANE_COMPONENT_arcane_dotnet_coreclr)
target_compile_definitions(arcane_dotnet_coreclr PRIVATE ARCANE_DOTNET_ROOT=\"${CoreClrEmbed_ROOT_PATH}\")
target_link_libraries(arcane_dotnet_coreclr PRIVATE arcane_core arcane_utils)
arcane_register_library(arcane_dotnet_coreclr OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
