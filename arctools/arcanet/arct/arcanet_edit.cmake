
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(arcanet_begin)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCT_PATH)
    set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
  endif ()

  if (DEFINED ${ARCANET_BEGIN_ON})
    message(FATAL_ERROR "'arcanet_begin()' has been already called. Call 'arcanet_end()' function to end it.")
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
endfunction()

# ----------------------------------------------------------------------------

function(arcanet_end)
  if (NOT DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arcanet_end()' cannot be called without a call to a 'arcanet_begin()' function.")
  endif ()
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' function to end it.")
  endif ()

  file(WRITE "${ARCANET_BEGIN_ON}" ${ARCANET_JSON})

  #

  unset(ARCANET_JSON PARENT_SCOPE)
  unset(ARCANET_BEGIN_ON PARENT_SCOPE)
  unset(ARCANET_JSON_PART_PATH PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arcanet_create_or_edit_general)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' function to end it.")
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

function(arcanet_create_or_edit_common name_common)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' function to end it.")
  endif ()

  # La partie config commons est dans :
  # {
  #   "commons": {
  #     "${name_common}": {
  #       // Config
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "commons" ${name_common} "")

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

function(arcanet_create_or_edit_case name_case)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' function to end it.")
  endif ()

  # La partie config cases est dans :
  # {
  #   "cases": {
  #     "${name_case}": {
  #       // Config
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "cases" ${name_case} "")

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

function(arcanet_create_or_edit_variation name_variation name_variant)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' function to end it.")
  endif ()

  # La partie config variations est dans :
  # {
  #   "variations": {
  #     "${name_variation}": {
  #       "${name_variant}": {
  #         // Config
  #       }
  #     }
  #   }
  # }
  set(ARCANET_JSON_PART_PATH "variations" ${name_variation} ${name_variant})

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

function(arcanet_create_or_edit_end)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
  endif ()

  list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
  list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)
  list(GET ARCANET_JSON_PART_PATH 2 ARCANET_JSON_PART_PATH_2)

  # Les autres fonctions ont rempli le bout de json ARCANET_JSON_PART. On doit le copier à l'adresse enregistrée par
  # les fonctions "arcanet_create_or_edit_X()"
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2} ${ARCANET_JSON_PART})

  #

  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
  unset(ARCANET_JSON_PART PARENT_SCOPE)
  unset(ARCANET_JSON_PART_PATH PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(arcanet_define_name display_name_case)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_add_dependency_a dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_add_dependency_b dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arcane_set_dataset dataset_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arcane_add_option option_name option_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arccher_define_mpi_path mpi_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arccher_add_envvar envvar_name envvar_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arccher_define_mpi_nb_procs nb_mpi)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

function(arcanet_arccher_define_executable exe_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' function.")
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

arcanet_begin(OVERWRITE_ARCT)

arcanet_create_or_edit_general()
arcanet_define_name("General")
arcanet_arcane_set_dataset("general_dataset.arc")
arcanet_arccher_define_mpi_path(mpi_path)
arcanet_arccher_add_envvar("ARCANE_USE_BACKWARDCPP" "1")
arcanet_create_or_edit_end()


arcanet_create_or_edit_common("4procs")
arcanet_define_name("4 procs")
arcanet_arccher_define_mpi_nb_procs(4)
arcanet_create_or_edit_end()


arcanet_create_or_edit_common("4threads")
arcanet_define_name("4 threads")
arcanet_arcane_add_option("S" "4")
arcanet_create_or_edit_end()


arcanet_create_or_edit_common("16mpithreads")
arcanet_add_dependency_b("4procs")
arcanet_add_dependency_b("4threads")
arcanet_create_or_edit_end()

arcanet_create_or_edit_variation("nb_iterations" "10")
arcanet_arcane_add_option("MaxIteration" "10")
arcanet_create_or_edit_end()

arcanet_create_or_edit_variation("nb_iterations" "20")
arcanet_arcane_add_option("MaxIteration" "20")
arcanet_create_or_edit_end()

arcanet_create_or_edit_variation("nb_iterations" "30")
arcanet_arcane_add_option("MaxIteration" "30")
arcanet_create_or_edit_end()

arcanet_create_or_edit_case("mon_test_1")
arcanet_define_name("Mon Test 1")
arcanet_add_dependency_b("nb_iterations!10")
arcanet_add_dependency_a("16mpithreads")
arcanet_arcane_set_dataset("truc.arc")
arcanet_arcane_add_option("MaxIteration" "3")
arcanet_arcane_add_option("MaxIteration" "4")
arcanet_arccher_define_executable(bin_path)
arcanet_arccher_add_envvar("VARIABLE" "VALUE")
arcanet_create_or_edit_end()

arcanet_create_or_edit_case("mon_test_2")
arcanet_define_name("Mon Test 2")
arcanet_arcane_set_dataset("truc.arc")
arcanet_arcane_add_option("MaxIteration" "3")
arcanet_arcane_add_option("MaxIteration" "4")
arcanet_arccher_define_executable(bin_path)
arcanet_arccher_add_envvar("VARIABLE" "VALUE")
arcanet_create_or_edit_end()

arcanet_end()
