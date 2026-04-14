
set(ARCANE_TEST_CASEPATH ${TEST_PATH}/material)

arcane_add_test_sequential(material1_simd1 testMaterialSimd-1.arc)

arcane_add_test(material_sync1 testMaterial-sync-1.arc)
arcane_add_test_parallel_thread(material_sync1 testMaterial-sync-1.arc 4)
arcane_add_test_parallel(material_sync2 testMaterial-sync-2.arc 4)
arcane_add_test_parallel(material_sync2_v3 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)
arcane_add_test(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
arcane_add_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
arcane_add_test_message_passing_hybrid(material_sync2_v7 CASE_FILE testMaterial-sync-2.arc NB_MPI 3 NB_SHM 4 ARGS -We,ARCANE_MATSYNCHRONIZE_VERSION,7)

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_accelerator_test_parallel(material_sync2_v6 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,6)
  arcane_add_accelerator_test_parallel(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_accelerator_test_parallel(material_sync2_v8 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,8)
  arcane_add_accelerator_test_parallel(material_sync2_vacc testMaterial-sync-2.arc 4 -We,ARCANE_ACC_MAT_SYNCHRONIZER,1)
  arcane_add_accelerator_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_accelerator_test_parallel_thread(material_sync2_vacc testMaterial-sync-2.arc 4 -We,ARCANE_ACC_MAT_SYNCHRONIZER,1)
endif ()

arcane_add_test(material_sync3 testMaterial-sync-3.arc)
arcane_add_test_parallel_thread(material_sync3 testMaterial-sync-3.arc 4)

#################################
# Material Heat
#################################

foreach(test_index 1 2 3 4)
  foreach(opt_level 0 3 7 11 15)
    set(TEST_MODIFICATION_FLAG ${opt_level})

    # Test sans équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "false")
    set(_TEST_FILENAME "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}.arc")
    configure_file("${ARCANE_TEST_CASEPATH}/testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME}")

    # Test avec équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "true")
    set(_TEST_FILENAME_LB "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}-lb.arc")
    configure_file("${ARCANE_TEST_CASEPATH}/testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME_LB}")
    if (ARCANE_HAS_ACCELERATOR_API)
      arcane_add_test_sequential(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME})
      arcane_add_test_parallel_thread(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME} 4)
      arcane_add_test(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB})
      arcane_add_test_parallel_thread(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB} 4)
    endif()
  endforeach()
endforeach()
if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(material_heat2_opt15_2small testMaterialHeat-2-small-opt15.arc "-We,ARCANE_DEBUG_MATERIAL_MODIFIER,2" "-We,ARCANE_PRINT_USELESS_TRANSFORMATION,1")
  arcane_add_test_sequential(material_heat2_opt15_2small_force_transform testMaterialHeat-2-small-opt15.arc "-We,ARCANE_MATERIAL_FORCE_TRANSFORM,1")
  arcane_add_test_sequential(material_heat2_accelerator
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc"
    "-We,ARCANE_MATERIALMNG_ADDITIONAL_CAPACITY_RATIO,0.5"
    "-m 20"
  )
  arcane_add_accelerator_test_sequential(material_heat2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat2_accelerator_noqueue
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_MATERIALMNG_USE_QUEUE,0")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_generic_copy
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,1")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_generic_copy_one_queue
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,2")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_force_multiple_resize
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_FORCE_MULTIPLE_COMMAND_FOR_MATERIAL_RESIZE,1")

  arcane_add_test_sequential(material_heat4_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat4_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init1_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20" "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,1")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20"
    "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,2"  "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,2")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init3_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20"
    "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,3" "-We,ARCANE_ALLENVCELL_FOR_RUNCOMMAND,1")
  if(HDF5_FOUND)
    arcane_add_test(material_heat2_vtkhdfv2 testMaterialHeat-2-vtkhdfv2.arc)
  endif()

endif()
#################################
#################################

# ----------------------------------------------------------------------------
# Test wrapping C#
# Il faut que le wrapper matériaux soit disponible
if (TARGET arcane_dotnet_wrapper_cea_materials)
  message(STATUS "Add target to test C# material wrapper: current_dir=${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "ArcaneSourceDir: ${Arcane_SOURCE_DIR}")
  include(${Arcane_SOURCE_DIR}/tools/wrapper/ArcaneSwigUtils.cmake)
  set_property(SOURCE material/MaterialEosService.i PROPERTY CPLUSPLUS ON)
  set_property(SOURCE material/MaterialEosService.i PROPERTY USE_SWIG_DEPENDENCIES TRUE)
  set_property(SOURCE material/MaterialEosService.i PROPERTY INCLUDE_DIRECTORIES
    ${Arccore_SOURCE_DIR}/src/base
    ${Arccore_SOURCE_DIR}/src/common
    ${Arcane_SOURCE_DIR}/src
    ${Arcane_SOURCE_DIR}/tools/wrapper
    ${Arcane_SOURCE_DIR}/ceapart/tools/wrapper)
  set_property(SOURCE material/MaterialEosService.i PROPERTY GENERATED_INCLUDE_DIRECTORIES
    ${Arcane_SOURCE_DIR}/src
  )
  set_property(SOURCE material/MaterialEosService.i PROPERTY COMPILE_OPTIONS -namespace MaterialEos)
  set(MATERIAL_CSHARP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
  file(MAKE_DIRECTORY ${MATERIAL_CSHARP_OUTDIR})

  # NOTE: le nom doit commencer par 'lib' pour que mono trouve la bibliothèque
  # au moment de l'exécution
  swig_add_library(test_material_eos_csharp
    TYPE SHARED
    LANGUAGE CSHARP
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/material/MaterialEosService.i
    OUTPUT_DIR ${MATERIAL_CSHARP_OUTDIR}/cs_files
    OUTFILE_DIR ${MATERIAL_CSHARP_OUTDIR}/cpp_files
  )
  # Génère une cible 'test_material_eos_csharp_swig_depend' pour garantir que le
  # wrapper swig est bien exécuté avant de compiler les tests.
  if (CMAKE_VERSION VERSION_LESS_EQUAL 3.21 OR CMAKE_GENERATOR MATCHES "Make")
    add_custom_target(test_material_eos_csharp_swig_depend ALL DEPENDS test_material_eos_csharp)
  else ()
    get_property(support_files TARGET test_material_eos_csharp PROPERTY SWIG_SUPPORT_FILES)
    add_custom_target(test_material_eos_csharp_swig_depend ALL DEPENDS "${support_files}")
  endif ()

  target_link_libraries(test_material_eos_csharp PUBLIC arcane_dotnet_wrapper_cea_materials arcane_materials)
  target_include_directories(test_material_eos_csharp PUBLIC ${Arcane_SOURCE_DIR}/tools/wrapper)
  # Il faut que la cible soit installée au même endroit que l'exécutable de test pour
  # qu'elle soit trouvée facilement.
  #set_property(TARGET libmaterial_eos_csharp PROPERTY LIBRARY_OUTPUT_DIRECTORY ${EOS_BINARY_DIR})
  target_link_libraries(arcane_tests_lib PUBLIC test_material_eos_csharp)
endif ()

#################################
#################################

include(material/TestDotNet.cmake)

#################################
#################################


if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test(material1 testMaterial-1.arc "-m 10" "-We,ARCANE_MATERIALSYNCHRONIZER_ACCELERATOR_MODE,1")
  arcane_add_test_parallel_thread(material1 testMaterial-1.arc 4 "-m 10" "-We,ARCANE_MATERIALSYNCHRONIZER_ACCELERATOR_MODE,1")
  arcane_add_test_parallel(material1_legacy_sync testMaterial-1.arc 4 "-m 10" "-We,ARCANE_MATERIAL_LEGACY_SYNCHRONIZE,1")
  if (LZ4_FOUND)
    arcane_add_test_sequential(material1_lz4 testMaterial-1.arc "-m 10" "-We,ARCANE_MATERIAL_DATA_COMPRESSOR_NAME,LZ4DataCompressor")
  endif ()
  arcane_add_test_sequential_task(material1 testMaterial-1.arc 4 "-m 10")

  # Les trois tests suivants doivent faire le même nombre d'itérations et avoir le
  # même nombre de mailles (pour test)
  ARCANE_ADD_TEST(material2 testMaterial-2.arc "-m 13")
  ARCANE_ADD_TEST(material2_opt1 testMaterial-2-opt1.arc "-m 13")
  ARCANE_ADD_TEST(material2_opt3 testMaterial-2-opt3.arc "-m 13")
  ARCANE_ADD_TEST(material2_opt5 testMaterial-2-opt5.arc "-m 13")
  ARCANE_ADD_TEST(material2_opt7 testMaterial-2-opt7.arc "-m 13")
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv2 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,2)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv3 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv6 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,6)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv7 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv8 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,8)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncvacc testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_ACC_MAT_SYNCHRONIZER,1)
  ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint testMaterial-checkpoint.arc 3 3)
  ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint_recreate testMaterial-checkpoint-recreate.arc 3 3)

  arcane_add_test_sequential_task(material2 testMaterial-2task.arc 4 "-m 1")


  ARCANE_ADD_TEST(material3 testMaterial-3.arc "-m 20")
  # NOTE Ajoute test optmisation uniquement en sequentiel car pour l'instant cela
  # ne marche pas en parallele a cause de la suppression de mailles.
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt1 testMaterial-3-opt1.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt3 testMaterial-3-opt3.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt5 testMaterial-3-opt5.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt7 testMaterial-3-opt7.arc "-m 20")
  if (NOT ARCANE_DISABLE_PERFCOUNTER_TESTS)
    arcane_add_test_sequential(material3_opt7_trace testMaterial-3-opt7.arc "-m 20" "-We,ARCANE_TRACE_ENUMERATOR,1")
  endif ()

  ARCANE_ADD_TEST_PARALLEL(material3_opt7_lb testMaterial-3-opt7-lb.arc 4 "-m 20")

endif ()

###################
# WRAPPING '.Net' #
###################
# TODO: Utiliser les macros génériques pour tester pour les modes
if (TARGET arcane_test_cs)
  # Cette variable est utilisée dans 'arcane_add_csharp_test_sequential'
  set(ARCANE_TEST_DOTNET_ASSEMBLY "${LIBRARY_OUTPUT_PATH}/ArcaneMaterialsTest.dll")
  message(STATUS "Adding '.Net' tests")
  if (MONO_EXEC)
    arcane_add_test_sequential(material2_cs testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} -m 4)
  endif ()
  if (TARGET arcane_dotnet_coreclr)
    arcane_add_test_sequential(material2_cs_coreclr testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} --dotnet-runtime=coreclr -m 4)
    arcane_add_test_sequential(material_eos_cs_coreclr testMaterial-eos-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} --dotnet-runtime=coreclr -m 4)
  endif ()
  arcane_add_csharp_test_sequential(material3_cs testMaterial-2-opt7-cs.arc -m 4 )
endif ()

