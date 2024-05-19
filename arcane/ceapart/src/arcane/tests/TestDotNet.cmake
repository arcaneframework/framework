# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion des tests C#.
# Il faut le wrapper C# pour pouvoir utiliser ces tests
if (NOT TARGET arcane_dotnet_wrapper_services)
  message(STATUS "Dont use '.Net' tests because 'services' wrapper is not available")
  return()
endif()
if (NOT TARGET arcane_dotnet_wrapper_cea_materials)
  message(STATUS "Dont use '.Net' tests because 'materials' wrapper is not available")
  return()
endif()
  
set(ARCANECEA_HAS_DOTNET_TESTS TRUE)
set(OUTLIB ${LIBRARY_OUTPUT_PATH}/ArcaneCeaTest.dll)
set(CSPATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CSOUTPATH ${CMAKE_CURRENT_BINARY_DIR})
configure_file(ArcaneCeaTest.csproj.in ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneCeaTest/ArcaneCeaTest.csproj @ONLY)

add_custom_command(OUTPUT ${CSOUTPATH}/MeshMaterialCSharpUnitTest_axl.cs
  DEPENDS ${CSPATH}/MeshMaterialCSharpUnitTest.axl ${ARCANE_AXL2CC} dotnet_axl_depend
  COMMAND ${ARCANE_AXL2CC}
  ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/MeshMaterialCSharpUnitTest.axl)

add_custom_target(arcanecea_test_cs_generate ALL DEPENDS
  ${CSOUTPATH}/MeshMaterialCSharpUnitTest_axl.cs
  )
arcane_add_global_csharp_target(arcanecea_test_cs
  BUILD_DIR ${LIBRARY_OUTPUT_PATH}
  ASSEMBLY_NAME ArcaneCeaTest.dll
  PROJECT_PATH ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneCeaTest
  PROJECT_NAME ArcaneCeaTest.csproj
  MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
  DOTNET_TARGET_DEPENDS dotnet_wrapper_cea_materials dotnet_wrapper_services
  DEPENDS
  arcanecea_test_cs_generate
  ${CSPATH}/Test1.cs
  ${CSPATH}/StiffenedGasMaterialEos.cs
  test_material_eos_csharp
  )
