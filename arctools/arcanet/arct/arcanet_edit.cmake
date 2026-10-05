
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(_arct_begin)
  if (NOT ARGS_ARCT_PATH)
    set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
  endif ()

  if (DEFINED ${ARCANET_BEGIN_ON})
    message(FATAL_ERROR "'arct_begin()' has been already called. Call 'arct_end()' function to end it.")
  endif ()

  if (EXISTS "${ARGS_ARCT_PATH}" AND NOT ARGS_OVERWRITE_ARCT)
    file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)
  else ()
    set(ARCANET_JSON "{}")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "{}")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "_" "0")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arcane" "0")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arccher" "0")
  endif ()

  #

  set(ARCANET_BEGIN_ON "${ARGS_ARCT_PATH}" PARENT_SCOPE)
  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

function(arct_begin)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  _arct_begin()
endfunction()

# ----------------------------------------------------------------------------

macro(_arct_end)
  if (NOT DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arct_end()' cannot be called without a call to a 'arct_begin()' function.")
  endif ()

  file(WRITE "${ARCANET_BEGIN_ON}" ${ARCANET_JSON})

  #

  unset(ARCANET_JSON PARENT_SCOPE)
  unset(ARCANET_BEGIN_ON PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

function(arct_end)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_X()' has been already called. Call 'arct_config_create_or_edit_end()' function to end it.")
  endif ()

  _arct_end()
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_create_or_edit_general)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_X()' has been already called. Call 'arct_config_create_or_edit_end()' function to end it.")
  endif ()

  if (NOT DEFINED ARCANET_BEGIN_ON)
    _arct_begin()
    set(ARCANET_WRITE_AT_END_PART "ON" PARENT_SCOPE)
  endif ()

  # La partie config general est dans :
  # {
  #   "general": {
  #     // Config
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "general" "" "")

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)

  string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
    set(ARCANET_JSON_PART "{}")
  endif ()

  #

  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
  set(ARCANET_JSON_PART_PATH "${ARCANET_JSON_PART_PATH}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_create_or_edit_common)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH NAME_COMMON)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_NAME_COMMON)
    message(FATAL_ERROR "Argument NAME_COMMON not defined")
  endif ()

  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_X()' has been already called. Call 'arct_config_create_or_edit_end()' function to end it.")
  endif ()

  if (NOT DEFINED ARCANET_BEGIN_ON)
    _arct_begin()
    set(ARCANET_WRITE_AT_END_PART "ON" PARENT_SCOPE)
  endif ()

  # La partie config commons est dans :
  # {
  #   "commons": {
  #     "${ARGS_NAME_COMMON}": {
  #       // Config
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "commons" ${ARGS_NAME_COMMON} "")

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
  list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)

  # Deux niveaux à créer un par un.
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
  endif ()
  string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
    set(ARCANET_JSON_PART "{}")
  endif ()

  #

  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
  set(ARCANET_JSON_PART_PATH "${ARCANET_JSON_PART_PATH}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_create_or_edit_case)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH NAME_CASE)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_NAME_CASE)
    message(FATAL_ERROR "Argument NAME_CASE not defined")
  endif ()

  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_X()' has been already called. Call 'arct_config_create_or_edit_end()' function to end it.")
  endif ()

  if (NOT DEFINED ARCANET_BEGIN_ON)
    _arct_begin()
    set(ARCANET_WRITE_AT_END_PART "ON" PARENT_SCOPE)
  endif ()

  # La partie config cases est dans :
  # {
  #   "cases": {
  #     "${ARGS_NAME_CASE}": {
  #       // Config
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "cases" ${ARGS_NAME_CASE} "")

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
  list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)

  # Deux niveaux à créer un par un.
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
  endif ()
  string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
    set(ARCANET_JSON_PART "{}")
  endif ()

  #

  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
  set(ARCANET_JSON_PART_PATH "${ARCANET_JSON_PART_PATH}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_create_or_edit_variation)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH NAME_VARIATION NAME_VARIANT)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_NAME_VARIATION)
    message(FATAL_ERROR "Argument NAME_VARIATION not defined")
  endif ()
  if (NOT ARGS_NAME_VARIANT)
    message(FATAL_ERROR "Argument NAME_VARIANT not defined")
  endif ()

  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_X()' has been already called. Call 'arct_config_create_or_edit_end()' function to end it.")
  endif ()

  if (NOT DEFINED ARCANET_BEGIN_ON)
    _arct_begin()
    set(ARCANET_WRITE_AT_END_PART "ON" PARENT_SCOPE)
  endif ()

  # La partie config variations est dans :
  # {
  #   "variations": {
  #     "${ARGS_NAME_VARIATION}": {
  #       "${ARGS_NAME_VARIANT}": {
  #         // Config
  #       }
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "variations" ${ARGS_NAME_VARIATION} ${ARGS_NAME_VARIANT})

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
  list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)
  list(GET ARCANET_JSON_PART_PATH 2 ARCANET_JSON_PART_PATH_2)

  # Trois niveaux à créer un par un.
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
  endif ()
  # ARCANET_JSON_PART contiendra la partie config du json.
  string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2} "{}")
    set(ARCANET_JSON_PART "{}")
  endif ()

  #

  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
  set(ARCANET_JSON_PART_PATH "${ARCANET_JSON_PART_PATH}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_create_or_edit_end)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
  list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)
  list(GET ARCANET_JSON_PART_PATH 2 ARCANET_JSON_PART_PATH_2)

  # Les autres fonctions ont rempli le bout de json ARCANET_JSON_PART. On doit le copier à l'adresse enregistrée par
  # les fonctions "arct_config_create_or_edit_X()"
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2} ${ARCANET_JSON_PART})



  #

  if (DEFINED ARCANET_WRITE_AT_END_PART)
    _arct_end()
    unset(ARCANET_WRITE_AT_END_PART PARENT_SCOPE)
  else ()
    set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  endif ()

  unset(ARCANET_JSON_PART PARENT_SCOPE)
  unset(ARCANET_JSON_PART_PATH PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(arct_config_define_name display_name_case)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "name" "\"${display_name_case}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_add_dependency_a dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" "depend_a")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_a" "[]")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_a" 999 "\"${dependency}\"" )

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_config_add_dependency_b dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" "depend_b")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_b" "[]")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_b" 999 "\"${dependency}\"" )

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(arct_arcane_set_dataset dataset_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "dataset" "\"${dataset_path}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_arcane_add_option option_name option_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane" "options")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "options" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "options" "${option_name}" "\"${option_value}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(arct_arccher_define_mpi_path mpi_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "mpi_binary" "\"${mpi_path}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_arccher_add_envvar envvar_name envvar_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher" "env_var")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "env_var" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "env_var" "${envvar_name}" "\"${envvar_value}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_arccher_define_mpi_nb_procs nb_mpi)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "mpi" "${nb_mpi}")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_arccher_define_executable exe_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "code_binary" "\"${exe_path}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

#arct_begin(OVERWRITE_ARCT)

arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_config_define_name("General")
arct_arcane_set_dataset("general_dataset.arc")
arct_arccher_define_mpi_path(mpi_path)
arct_arccher_add_envvar("ARCANE_USE_BACKWARDCPP" "1")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4procs")
arct_config_define_name("4 procs")
arct_arccher_define_mpi_nb_procs(4)
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "4threads")
arct_config_define_name("4 threads")
arct_arcane_add_option("S" "4")
arct_config_create_or_edit_end()


arct_config_create_or_edit_common(NAME_COMMON "16mpithreads")
arct_config_add_dependency_b("4procs")
arct_config_add_dependency_b("4threads")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "10")
arct_arcane_add_option("MaxIteration" "10")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "20")
arct_arcane_add_option("MaxIteration" "20")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_iterations" NAME_VARIANT "30")
arct_arcane_add_option("MaxIteration" "30")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_1")
arct_config_define_name("Mon Test 1")
arct_config_add_dependency_b("nb_iterations!10")
arct_config_add_dependency_a("16mpithreads")
arct_arcane_set_dataset("truc.arc")
arct_arcane_add_option("MaxIteration" "3")
arct_arcane_add_option("MaxIteration" "4")
arct_arccher_define_executable(bin_path)
arct_arccher_add_envvar("VARIABLE" "VALUE")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "mon_test_2")
arct_config_define_name("Mon Test 2")
arct_arcane_set_dataset("truc.arc")
arct_arcane_add_option("MaxIteration" "3")
arct_arcane_add_option("MaxIteration" "4")
arct_arccher_define_executable(bin_path)
arct_arccher_add_envvar("VARIABLE" "VALUE")
arct_config_create_or_edit_end()

#arct_end()
