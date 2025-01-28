# Avant d'inclure ce fichier, il faut positionner les variables CMake suivantes:
#   ARCANE_TEST_DRIVER: nom du driver (normalement dans le cache)
#   ARCANE_TEST_WORKDIR: répertoire de travail pour les tests.
#   ARCANE_TEST_CASEPATH: répertoire contenant les JDD.
# La variable suivante est optionnelle:
#   ARCANE_TEST_EXECNAME: nom de l'exécutable (arcane_tests_exec par défaut)
#   ARCANE_TEST_DOTNET_ASSEMBLY: nom de l'assembly '.Net' a charger (ArcaneTest.dll par défaut)

set(ARCANE_TEST_LAUNCH_COMMAND ${ARCANE_TEST_DRIVER} launch)
set(ARCANE_TEST_SCRIPT_COMMAND ${ARCANE_TEST_DRIVER} script)
if (ARCANE_TEST_EXECNAME)
  message(STATUS "[Tests] Using executable '${ARCANE_TEST_EXECNAME}' for tests")
  set(ARCANE_TEST_LAUNCH_COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -E ${ARCANE_TEST_EXECNAME})
endif()
if (ARCANE_TEST_DOTNET_ASSEMBLY)
  message(STATUS "[Tests] Using '.Net' assembly '${ARCANE_TEST_DOTNET_ASSEMBLY}' for tests")
endif()

# Cherche le chemin du test et le retourne dans 'full_case_file'
macro(ARCANE_GET_CASE_PATH case_file)
  string(REGEX MATCH "^/" _is_full ${case_file})
  if(_is_full)
    if(VERBOSE)
      message(STATUS "      IS FULL! ${case_file}")
    endif(VERBOSE)
    set(full_case_file ${case_file})
  else(_is_full)
    set(full_case_file ${ARCANE_TEST_CASEPATH}/${case_file})
  endif(_is_full)
endmacro(ARCANE_GET_CASE_PATH case_file)

# ----------------------------------------------------------------------------
# Fonction Arcane pour ajouter un cas test
#
# Encapsule la fonction 'add_test' de CMake pour ajouter certaines informations
# utiles pour les tests (comme par exemple une variable d'environnement contenant
# le nom du test).
function(arcane_add_test_direct)
  set(options        )
  set(oneValueArgs   NAME WORKING_DIRECTORY)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (NOT ARGS_NAME)
    message(FATAL_ERROR "No 'NAME' argument")
  endif()
  if (NOT ARGS_WORKING_DIRECTORY)
    message(FATAL_ERROR "No 'WORKING_DIRECTORY' argument")
  endif()
  if (NOT ARGS_COMMAND)
    message(FATAL_ERROR "No 'COMMAND' argument")
  endif()
  add_test(NAME ${ARGS_NAME}
    COMMAND ${ARGS_COMMAND}
    WORKING_DIRECTORY ${ARGS_WORKING_DIRECTORY})
  set(RESULT_FILE "${Arcane_SOURCE_DIR}/tests/results/${ARGS_NAME}.txt")
  if (EXISTS "${RESULT_FILE}")
    message(VERBOSE "ADD_TEST_RESULT_FILE name=${ARGS_NAME} path=${RESULT_FILE}")
    set_property(TEST ${ARGS_NAME} APPEND PROPERTY ENVIRONMENT "ARCANE_TEST_RESULT_FILE=${RESULT_FILE}")
  endif()
  set_property(TEST ${ARGS_NAME} APPEND PROPERTY ENVIRONMENT "ARCANE_TEST_NAME=${ARGS_NAME}")
endfunction()

# ----------------------------------------------------------------------------
# Ajoute un test sequentiel
function(ARCANE_ADD_TEST_SEQUENTIAL test_name case_file)
  if(VERBOSE)
    message(STATUS "    ADD TEST SEQUENTIEL OPT=${ARCANE_TEST_CASEPATH} ${case_file}")
  endif()
  ARCANE_GET_CASE_PATH(${case_file})
  arcane_add_test_direct(NAME ${test_name}
    COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} ${ARGN} ${full_case_file}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
endfunction()

# Ajoute un test sequentiel avec reprise
function(ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL test_name case_file nb_continue nb_iteration)
  if(VERBOSE)
    message(STATUS "    ADD TEST CHECKPOINT SEQUENTIEL OPT=${ARCANE_TEST_CASEPATH} ${case_file}")
  endif()
  ARCANE_GET_CASE_PATH(${case_file})
  arcane_add_test_direct(NAME ${test_name}
    COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -c ${nb_continue} -m ${nb_iteration} ${ARGN} ${full_case_file}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
endfunction()

# Ajoute un test sequentiel avec variable d'environnement
function(ARCANE_ADD_TEST_SEQUENTIAL_ENV test_name case_file envvar envvalue)
  if(VERBOSE)
    message(STATUS "    ADD TEST SEQUENTIEL OPT=${ARCANE_TEST_CASEPATH} ${case_file} ENV:${envar}=${envvalue}")
  endif()
  arcane_add_test_direct(NAME ${test_name}
    COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -We,${envvar},${envvalue} ${ARGN} ${ARCANE_TEST_CASEPATH}/${case_file}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
endfunction()

# Ajoute un test parallele
macro(ARCANE_ADD_TEST_PARALLEL test_name case_file nb_proc)
  if(TARGET arcane_mpi)
    if(VERBOSE)
      MESSAGE(STATUS "    ADD TEST PARALLEL MPI OPT=${test_name} ${case_file}")
    endif()
    ARCANE_GET_CASE_PATH(${case_file})
    arcane_add_test_direct(NAME ${test_name}_${nb_proc}proc
      COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -n ${nb_proc} ${ARGN} ${full_case_file}
      WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
    set_tests_properties(${test_name}_${nb_proc}proc PROPERTIES PROCESSORS ${nb_proc})
  endif()
endmacro()

# Ajoute un test parallele
macro(arcane_add_test_sequential_task test_name case_file nb_task)
  # Si nb_task vaut 0, on utilise tous les CPU du noeud.
  if(ARCANE_HAS_TASKS)
    if(VERBOSE)
      message(STATUS "    Add Test Task OPT=${test_name} ${case_file}")
    endif()
    set(_TEST_NAME ${test_name}_task${nb_task})
    if (${nb_task} STREQUAL 0)
      set(_TEST_NAME ${test_name}_taskmax)
    endif()
    arcane_get_case_path(${case_file})
    arcane_add_test_direct(NAME ${_TEST_NAME}
      COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -K ${nb_task} ${ARGN} ${full_case_file}
      WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
    if (${nb_task} STREQUAL 0)
      set_tests_properties(${_TEST_NAME} PROPERTIES RUN_SERIAL TRUE)
    else()
      set_tests_properties(${_TEST_NAME} PROPERTIES PROCESSORS ${nb_task})
    endif()
  endif()
endmacro()

# Ajoute un test parallele en mode mémoire partagée
macro(arcane_add_test_parallel_thread test_name case_file nb_proc)
  if(NOT ARCANE_USE_MPC)
    if(VERBOSE)
      message(STATUS "    ADD TEST THREAD OPT=${test_name} ${case_file}")
    endif()
    ARCANE_GET_CASE_PATH(${case_file})
    arcane_add_test_direct(NAME ${test_name}_${nb_proc}thread
      COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -T ${nb_proc} ${ARGN} ${full_case_file}
      WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
    set_tests_properties(${test_name}_${nb_proc}thread PROPERTIES PROCESSORS ${nb_proc})
  endif()
endmacro()


# ----------------------------------------------------------------------------
# Ajoute un test message_passing en mode hybride (MPI+SHM).
# Le test n'est ajouté que si le mode hybride est disponible.
#
# Usage:
#
# arcane_add_test_message_passing_hybrid(test_name
#    [CASE_FILE case_file]
#    NB_MPI nb_mpi
#    NB_SHM nb_shm
#    [ARGS args]
# )
function(arcane_add_test_message_passing_hybrid test_name)
  set(options        )
  set(oneValueArgs   NB_MPI NB_SHM CASE_FILE)
  set(multiValueArgs ARGS)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(VERBOSE "    ADD TEST1 MessagePassing Hybrid OPT=${test_name} ${ARGS_CASE_NAME} nb_mpi=${ARGS_NB_MPI} nb_shm=${ARGS_NB_SHM}")

  if (NOT ARGS_NB_MPI)
    message(FATAL_ERROR "Missing argument 'NB_MPI'")
  endif()
  if (NOT ARGS_NB_SHM)
    message(FATAL_ERROR "Missing argument 'NB_SHM'")
  endif()
  set(nb_proc ${ARGS_NB_MPI})
  set(nb_thread ${ARGS_NB_SHM})

  if(NOT TARGET arcane_mpithread)
    return()
  endif()
  # L'argument CASE_FILE peut être nul
  if (ARGS_CASE_FILE)
    arcane_get_case_path(${ARGS_CASE_FILE})
  else()
    set(full_case_file "")
  endif()
  set(_arcane_test_name ${test_name}_hybrid_${nb_proc}_${nb_thread}_mpithread)
  arcane_add_test_direct(NAME ${_arcane_test_name}
    COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -n ${nb_proc} -T ${nb_thread} ${ARGS_ARGS} ${full_case_file}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
  # Calcule le nombre de coeurs pour ce test
  math(EXPR _total_nb_proc "${nb_thread} * ${nb_proc}")
  set_tests_properties(${_arcane_test_name} PROPERTIES PROCESSORS ${nb_proc})
  # Ajoute un lable 'LARGE_HYBRID' si le test dépasse 4 PE. Cela permet de le désactiver
  # Dans certains workflows du CI
  if (${_total_nb_proc} GREATER "4")
    set_tests_properties(${_arcane_test_name} PROPERTIES LABELS LARGE_HYBRID)
  endif()
endfunction()


# ----------------------------------------------------------------------------
# Fonction générique pour ajouter un test pour un ou plusieurs modes d'échange
# de message. Les tests ne sont ajoutés que si le mode d'échange de message
# concerné est disponible.
#
# Usage:
#
# arcane_add_test_message_generic(test_name
#    [CASE_FILE case_file]
#    [NB_MPI nb_mpi]
#    [NB_SHM nb_shm]
#    [ARGS args]
#    [MP_SEQUENTIAL]
#    [MP_SHM]
#    [MP_MPI]
#    [MP_HYBRID]
#
# Si une des valeurs MP_SEQUENTIAL, MP_SHM, MP_MPI ou MPI_HYBRID est spécifiée,
# alors le test correspond sera exécuté. Sinon on exécute le test les 4
# modes d'échange de message. Si NB_MPI n'est pas spécifié, il aura pour valeur 4.
# Si NB_SHM n'est pas spécifié, il aura pour valeur 3.
# )

function(arcane_add_test_generic test_name)
  set(options        MP_SEQUENTIAL MP_SHM MP_MPI MP_HYBRID)
  set(oneValueArgs   TYPE NB_MPI NB_SHM CASE_FILE)
  set(multiValueArgs ARGS)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(VERBOSE "    ADD GENERIC TEST Hybrid OPT=${test_name} ${ARGS_CASE_NAME} nb_mpi=${ARGS_NB_MPI} nb_shm=${ARGS_NB_SHM} type=${ARGS_TYPE}")

  set(HAS_MP_VALUE FALSE)

  # Si un des types de 'message passing' est spécifié, on l'utilise.
  # Sinon on considère qu'on utilise toutes les mécanismes de message passing disponibles
  if (ARGS_MP_SEQUENTIAL OR ARGS_MP_SHM OR ARGS_MP_MPI OR ARGS_MP_HYBRID)
    set(HAS_MP_VALUE TRUE)
  else()
    set(ARGS_MP_SEQUENTIAL TRUE)
    set(ARGS_MP_SHM TRUE)
    set(ARGS_MP_MPI TRUE)
    set(ARGS_MP_HYBRID TRUE)
  endif()

  if (NOT ARGS_NB_MPI)
    set(ARGS_NB_MPI 4)
  endif()
  if (NOT ARGS_NB_SHM)
    set(ARGS_NB_SHM 3)
  endif()

  if (ARGS_MP_SEQUENTIAL)
    arcane_add_test_sequential(${test_name} ${ARGS_CASE_FILE} ${ARGS_ARGS})
  endif()
  if (ARGS_MP_MPI)
    arcane_add_test_parallel(${test_name} ${ARGS_CASE_FILE} ${ARGS_NB_MPI} ${ARGS_ARGS})
  endif()
  if (ARGS_MP_SHM)
    arcane_add_test_parallel_thread(${test_name} ${ARGS_CASE_FILE} ${ARGS_NB_SHM} ${ARGS_ARGS})
  endif()
  if (ARGS_MP_HYBRID)
    arcane_add_test_message_passing_hybrid(${test_name} CASE_FILE ${ARGS_CASE_FILE} NB_MPI ${ARGS_NB_MPI} NB_SHM ${ARGS_NB_SHM} ARGS ${ARGS_ARGS})
  endif()
endfunction()

# ----------------------------------------------------------------------------

# Ajoute un test parallele avec MPI+threads
macro(ARCANE_ADD_TEST_PARALLEL_MPITHREAD test_name case_file nb_proc nb_thread)
  arcane_add_test_message_passing_hybrid(${test_name} CASE_FILE ${case_file} NB_SHM ${nb_thread} NB_MPI ${nb_proc} ARGS ${ARGN})
endmacro()

# Ajoute un test avec tous les mécanismes d'échange de message
macro(arcane_add_test_parallel_all test_name case_file nb_proc1 nb_proc2)
  ARCANE_ADD_TEST(${test_name} ${case_file} ${ARGN})
  ARCANE_ADD_TEST_PARALLEL_THREAD(${test_name} ${case_file} ${nb_proc2} ${ARGN})
  arcane_add_test_message_passing_hybrid(${test_name} CASE_FILE ${case_file} NB_MPI ${nb_proc1} NB_SHM ${nb_proc2} ARGS ${ARGN})
endmacro()

# Ajoute un test parallele avec reprise
function(ARCANE_ADD_TEST_CHECKPOINT_PARALLEL test_name case_file nb_proc nb_continue nb_iteration)
  if(TARGET arcane_mpi)
    if(VERBOSE)
      MESSAGE(STATUS "    ADD TEST CHECKPOINT PARALLEL MPI OPT=${test_name} ${case_file}")
    endif()
    ARCANE_GET_CASE_PATH(${case_file})
    arcane_add_test_direct(NAME ${test_name}_${nb_proc}proc
      COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -c ${nb_continue} -m ${nb_iteration} -n ${nb_proc} ${ARGN} ${full_case_file}
      WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
    set_tests_properties(${test_name}_${nb_proc}proc PROPERTIES PROCESSORS ${nb_proc})
  endif()
endfunction()

# Ajoute un test parallele avec variable d'environnement
macro(ARCANE_ADD_TEST_PARALLEL_ENV test_name case_file nb_proc envvar envvalue)
  if(TARGET arcane_mpi)
    if(VERBOSE)
      message(STATUS "    ADD TEST PARALLEL MPI OPT=${test_name} ${case_file} ENV:${envvar}=${envvalue}")
    endif()
    arcane_add_test_direct(NAME ${test_name}_${nb_proc}proc
      COMMAND ${ARCANE_TEST_LAUNCH_COMMAND} -We,${envvar},${envvalue} -n ${nb_proc} ${ARGN} ${ARCANE_TEST_CASEPATH}/${case_file}
      WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR})
  endif()
endmacro()

# Ajoute un test sequentiel et parallele
macro(ARCANE_ADD_TEST test_name case_file)
  ARCANE_ADD_TEST_SEQUENTIAL(${test_name} ${case_file} ${ARGN})
  ARCANE_ADD_TEST_PARALLEL(${test_name} ${case_file} 4 ${ARGN})
endmacro()

# Ajoute un test sequentiel et parallele avec reprise
# test_name: nom du cas
# case_file: jeu de donnees
# nb_continue: nombre de reprises
# nb_iteration: nombre d'iteration pour chaque execution
macro(ARCANE_ADD_TEST_CHECKPOINT test_name case_file nb_continue nb_iteration)
  ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(${test_name} ${case_file} ${nb_continue} ${nb_iteration} ${ARGN})
  ARCANE_ADD_TEST_CHECKPOINT_PARALLEL(${test_name} ${case_file} 4 ${nb_continue} ${nb_iteration} ${ARGN})
endmacro()

macro(arcane_add_test_script test_name script_file)
  set(_TEST_NAME ${test_name})
  configure_file(${ARCANE_TEST_CASEPATH}/${script_file} ${CMAKE_CURRENT_BINARY_DIR}/${script_file} @ONLY)
  arcane_add_test_direct(NAME ${test_name}
    COMMAND ${ARCANE_TEST_SCRIPT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/${script_file}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
    )
endmacro()

# ----------------------------------------------------------------------------
# Ajoute un test séquentiel en C#
# Pour chaque test, on génère 4 variantes:
# - le lancement avec 'coreclr' via dotnet (coreclr_dotnet)
# - le lancement avec l'exécutable classique et coreclr intégré (coreclr_embedded)
#
function(arcane_add_csharp_test_direct)
  set(options        )
  set(oneValueArgs CASE_FILE_PATH TEST_NAME WORKING_DIRECTORY ASSEMBLY EXTERNAL_ASSEMBLY)
  set(multiValueArgs LAUNCH_COMMAND ARGS)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_TEST_NAME)
    message(FATAL_ERROR "No 'TEST_NAME' argument")
  endif()

  cmake_path(GET ARGS_CASE_FILE_PATH EXTENSION LAST_ONLY _EXTENSION_ARC)

  # Si l'extension du fichier est .in, alors on a un EXTERNAL_ASSEMBLY à situer.
  # On récupère le nom du fichier sans extension.
  # Sinon, on garde le répertoire du .arc d'origine.
  if (_EXTENSION_ARC STREQUAL ".in")
    cmake_path(GET ARGS_CASE_FILE_PATH STEM _FILENAME)
  else ()
    set(_FILE_ARC ${ARGS_CASE_FILE_PATH})
  endif ()

  if (ARGS_ASSEMBLY)
    set(_ALL_ARGS "--dotnet-assembly=${ARGS_ASSEMBLY}")
  endif ()

  if (DOTNET_EXEC)
    # Test avec coreclr direct
    if (ARGS_EXTERNAL_ASSEMBLY)
      set(_OUTPUT_DLL ${ARGS_WORKING_DIRECTORY}/${ARGS_EXTERNAL_ASSEMBLY}_coreclr.dll)
      if (_EXTENSION_ARC STREQUAL ".in")
        set(_FILE_ARC ${ARGS_WORKING_DIRECTORY}/${_FILENAME}_coreclr.arc)
        # Dans ce fichier, il n'y a que le _OUTPUT_DLL à définir.
        configure_file(${ARGS_CASE_FILE_PATH} ${_FILE_ARC} @ONLY)
      endif ()
      set(_EXTERNAL_ARGS "--dotnet-compile=${TEST_PATH}/${ARGS_EXTERNAL_ASSEMBLY}.cs")
      list(APPEND _EXTERNAL_ARGS "--dotnet-output-dll=${_OUTPUT_DLL}")
    endif ()

    arcane_add_test_direct(NAME ${ARGS_TEST_NAME}_coreclr_dotnet
      COMMAND ${ARGS_LAUNCH_COMMAND} -Z --dotnet-runtime=coreclr ${ARGS_ARGS} ${_FILE_ARC} ${_ALL_ARGS} ${_EXTERNAL_ARGS}
      WORKING_DIRECTORY ${ARGS_WORKING_DIRECTORY})
  endif()

  if (TARGET arcane_dotnet_coreclr)
    # Test avec coreclr embedded
    if (ARGS_EXTERNAL_ASSEMBLY)
      set(_OUTPUT_DLL ${ARGS_WORKING_DIRECTORY}/${ARGS_EXTERNAL_ASSEMBLY}_coreclr_embedded.dll)
      if (_EXTENSION_ARC STREQUAL ".in")
        set(_FILE_ARC ${ARGS_WORKING_DIRECTORY}/${_FILENAME}_coreclr_embedded.arc)
        configure_file(${ARGS_CASE_FILE_PATH} ${_FILE_ARC} @ONLY)
      endif ()
      set(_EXTERNAL_ARGS "--dotnet-compile=${TEST_PATH}/${ARGS_EXTERNAL_ASSEMBLY}.cs")
      list(APPEND _EXTERNAL_ARGS "--dotnet-output-dll=${_OUTPUT_DLL}")
    endif ()

    arcane_add_test_direct(NAME ${ARGS_TEST_NAME}_coreclr_embedded
      COMMAND ${ARGS_LAUNCH_COMMAND} -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-runtime=coreclr ${ARGS_ARGS} ${_FILE_ARC} ${_ALL_ARGS} ${_EXTERNAL_ARGS}
      WORKING_DIRECTORY ${ARGS_WORKING_DIRECTORY})
  endif()

endfunction()

# ----------------------------------------------------------------------------
# Ajoute un test séquentiel en C#
#
macro(arcane_add_csharp_test_sequential test_name case_file)
  if(VERBOSE)
    message(STATUS "    ADD C# SEQUENTIAL TEST=${ARCANE_TEST_CASEPATH} ${case_file}")
  endif()
  arcane_get_case_path(${case_file})
  message(STATUS "ADD C# test name=${test_name} assembly=${ARCANE_TEST_DOTNET_ASSEMBLY}")
  arcane_add_csharp_test_direct(TEST_NAME ${test_name}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
    LAUNCH_COMMAND ${ARCANE_TEST_LAUNCH_COMMAND}
    CASE_FILE_PATH ${full_case_file}
    ASSEMBLY ${ARCANE_TEST_DOTNET_ASSEMBLY}
    ARGS ${ARGN}
    )
endmacro()

# ----------------------------------------------------------------------------
# Ajoute un test séquentiel en C# avec chargement d'une dll externe.
#
macro(arcane_add_csharp_test_sequential_external_assembly test_name case_file assembly_file)
  if (VERBOSE)
    message(STATUS "    ADD C# SEQUENTIAL TEST=${ARCANE_TEST_CASEPATH} ${case_file} WITH EXTERNAL ASSEMBLY=${assembly_file}")
  endif ()
  arcane_get_case_path(${case_file})
  message(STATUS "ADD C# test name=${test_name} assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} external assembly=${assembly_file}")
  arcane_add_csharp_test_direct(
          TEST_NAME ${test_name}
          WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
          LAUNCH_COMMAND ${ARCANE_TEST_LAUNCH_COMMAND}
          CASE_FILE_PATH ${full_case_file}
          ASSEMBLY ${ARCANE_TEST_DOTNET_ASSEMBLY}
          EXTERNAL_ASSEMBLY ${assembly_file}
          ARGS ${ARGN}
  )
endmacro()

# ----------------------------------------------------------------------------
# Ajoute un test parallèle en C#
#
macro(arcane_add_csharp_test_parallel test_name case_file nb_proc)
  if(VERBOSE)
    message(STATUS "    ADD C# PARALELL TEST=${ARCANE_TEST_CASEPATH} ${case_file}")
  endif()
  set(_TEST_BASE_NAME ${test_name}_${nb_proc}proc)
  message(STATUS "ADD C# test name=${test_name} assembly=${ARCANE_TEST_DOTNET_ASSEMBLY}")
  arcane_get_case_path(${case_file})
  arcane_add_csharp_test_direct(TEST_NAME ${_TEST_BASE_NAME}
    WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
    LAUNCH_COMMAND ${ARCANE_TEST_LAUNCH_COMMAND}
    CASE_FILE_PATH ${full_case_file}
    ASSEMBLY ${ARCANE_TEST_DOTNET_ASSEMBLY}
    ARGS -n ${nb_proc} ${ARGN}
    )
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(ARCANE_ACCELERATOR_RUNTIME_NAME ${ARCANE_ACCELERATOR_RUNTIME})

# ----------------------------------------------------------------------------
# Ajoute un test séquentiel pour accélerateur si disponible
macro(arcane_add_accelerator_test_sequential test_name case_file)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME)
    message(STATUS "ADD ACCELERATOR test name=${test_name}")
    arcane_add_test_sequential(${test_name}_${ARCANE_ACCELERATOR_RUNTIME_NAME} ${case_file}
      "-A,UseAccelerator=1" ${ARGN}
      )
  endif()
endmacro()

# Ajoute un test parallèle MPI pour accélerateur si disponible
macro(arcane_add_accelerator_test_parallel test_name case_file nb_proc)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME)
    arcane_add_test_parallel(${test_name}_${ARCANE_ACCELERATOR_RUNTIME_NAME} ${case_file} ${nb_proc}
      "-A,AcceleratorRuntime=${ARCANE_ACCELERATOR_RUNTIME_NAME}" ${ARGN}
      )
  endif()
endmacro()

# Ajoute un test parallèle 'sharedmemory' pour accélerateur si disponible
macro(arcane_add_accelerator_test_parallel_thread test_name case_file nb_proc)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME)
    arcane_add_test_parallel_thread(${test_name}_${ARCANE_ACCELERATOR_RUNTIME_NAME} ${case_file} ${nb_proc}
      "-A,AcceleratorRuntime=${ARCANE_ACCELERATOR_RUNTIME_NAME}" ${ARGN}
      )
  endif()
endmacro()

# Ajoute un test parallèle 'hybride' pour accélerateur si disponible
macro(arcane_add_accelerator_test_message_passing_hybrid test_name)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME)
    arcane_add_test_message_passing_hybrid(${test_name}_${ARCANE_ACCELERATOR_RUNTIME_NAME} ${ARGN}
      "-A,AcceleratorRuntime=${ARCANE_ACCELERATOR_RUNTIME_NAME}"
    )
  endif()
endmacro()

macro(arcane_add_test_sequential_host_and_accelerator test_name case_file)
  arcane_add_test_sequential(${test_name} ${case_file} ${ARGN})
  arcane_add_test_sequential_task(${test_name} ${case_file} 4 ${ARGN})
  arcane_add_accelerator_test_sequential(${test_name} ${case_file} ${ARGN})
endmacro()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
