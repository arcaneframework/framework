include(srcs.cmake)

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    AdiProjectionModule.cc
    ${ARCANE_MATERIAL_SOURCES}
  )
  set(ARCANE_ACCELERATOR_SOURCES
    AdiProjectionModule.cc
    CartesianMeshTestUtils.cc
    MeshMaterialSyncUnitTest.cc
    MaterialHeatTestModule.cc
  )
  arcane_accelerator_add_source_files(${ARCANE_ACCELERATOR_SOURCES})
endif()

arcane_add_library(arcane_cea_tests
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/tests
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

target_compile_definitions(arcane_cea_tests PRIVATE ARCANE_COMPONENT_arcane_cea_tests)

if (TARGET arcane_materials)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_materials)
endif()
target_link_libraries(arcane_cea_tests PUBLIC arcane_cea_geometric arcane_cea ${ARCANE_BASE_LIBRARIES})
if(TARGET arcane_aleph)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph)
endif()
if(TARGET arcane_aleph_sloop)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph_sloop)
endif()
target_link_libraries(arcane_cea_tests PRIVATE $<TARGET_NAME_IF_EXISTS:arcane_accelerator>)

# ----------------------------------------------------------------------------
# Test wrapping C#
# Il faut que le wrapper matériaux soit disponible
if (TARGET arcane_dotnet_wrapper_cea_materials)
  message(STATUS "Add target to test C# material wrapper: current_dir=${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "ArcaneSourceDir: ${Arcane_SOURCE_DIR}")
  include(${Arcane_SOURCE_DIR}/tools/wrapper/ArcaneSwigUtils.cmake)
  set_property(SOURCE MaterialEosService.i PROPERTY CPLUSPLUS ON)
  set_property(SOURCE MaterialEosService.i PROPERTY USE_SWIG_DEPENDENCIES TRUE)
  set_property(SOURCE MaterialEosService.i PROPERTY INCLUDE_DIRECTORIES
    ${Arccore_SOURCE_DIR}/src/base
    ${Arcane_SOURCE_DIR}/src
    ${Arcane_SOURCE_DIR}/tools/wrapper
    ${Arcane_SOURCE_DIR}/ceapart/tools/wrapper)
  set_property(SOURCE MaterialEosService.i PROPERTY GENERATED_INCLUDE_DIRECTORIES
    ${Arcane_SOURCE_DIR}/ceapart/src
    )
  set_property(SOURCE MaterialEosService.i PROPERTY COMPILE_OPTIONS -namespace MaterialEos)
  set(MATERIAL_CSHARP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
  file(MAKE_DIRECTORY ${MATERIAL_CSHARP_OUTDIR})

  # NOTE: le nom doit commencer par 'lib' pour que mono trouve la bibliothèque
  # au moment de l'exécution
  swig_add_library(test_material_eos_csharp
    TYPE SHARED
    LANGUAGE CSHARP
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/MaterialEosService.i
    OUTPUT_DIR ${MATERIAL_CSHARP_OUTDIR}/cs_files
    OUTFILE_DIR ${MATERIAL_CSHARP_OUTDIR}/cpp_files
  )
  # Génère une cible 'test_material_eos_csharp_swig_depend' pour garantir que le
  # wrapper swig est bien exécuté avant de compiler les tests.
  if (CMAKE_VERSION VERSION_LESS_EQUAL 3.21 OR CMAKE_GENERATOR MATCHES "Make")
    add_custom_target(test_material_eos_csharp_swig_depend ALL DEPENDS test_material_eos_csharp)
  else()
    get_property(support_files TARGET test_material_eos_csharp PROPERTY SWIG_SUPPORT_FILES)
    add_custom_target(test_material_eos_csharp_swig_depend ALL DEPENDS "${support_files}")
  endif()

  target_link_libraries(test_material_eos_csharp PUBLIC arcane_dotnet_wrapper_cea_materials arcane_materials)
  target_include_directories(test_material_eos_csharp PUBLIC ${Arcane_SOURCE_DIR}/tools/wrapper)
  # Il faut que la cible soit installée au même endroit que l'exécutable de test pour
  # qu'elle soit trouvée facilement.
  #set_property(TARGET libmaterial_eos_csharp PROPERTY LIBRARY_OUTPUT_DIRECTORY ${EOS_BINARY_DIR})
  target_link_libraries(arcane_tests_lib PUBLIC test_material_eos_csharp)
endif()

# ----------------------------------------------------------------------------

# Indique que la bibliothèque des tests a besoin de celle-ci
target_link_libraries(arcane_tests_lib PUBLIC arcane_cea_tests)

arcane_register_library(arcane_cea_tests OPTIONAL)

include(TestDotNet.cmake)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
