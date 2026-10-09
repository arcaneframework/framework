
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(_arct_begin)
  if (NOT ARGS_ARCT_PATH)
    if(DEFINED ARCT_GLOBAL_ARCT_PATH)
      set(ARGS_ARCT_PATH "${ARCT_GLOBAL_ARCT_PATH}")
    else ()
      set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
    endif ()
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

macro(_arct_config_append_var type variation elem_to_add)

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Variation '${variation}' not found.")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" ${type})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Variation '${variation}' not found.")
  endif ()

  string(JSON ARCANET_ARRAY_LENGTH ERROR_VARIABLE ARCANET_ERROR LENGTH ${ARCANET_GET})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    set(ARCANET_ARRAY_LENGTH 0)
  endif ()

  if (${ARCANET_ARRAY_LENGTH} GREATER 0)
    math(EXPR ARCANET_ARRAY_LENGTH "${ARCANET_ARRAY_LENGTH} - 1")

    foreach (VAR_I RANGE ${ARCANET_ARRAY_LENGTH})

      string(JSON ARCANET_GET_VARIANT_NAME ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_GET} ${VAR_I})
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error")
      endif ()

      string(FIND ${ARCANET_GET_VARIANT_NAME} ${variation} ARCANET_POS_ELEM)

      if(NOT ARCANET_POS_ELEM EQUAL -1)
        string(APPEND ARCANET_GET_VARIANT_NAME ${elem_to_add})
        string(JSON ARCANET_GET SET ${ARCANET_GET} ${VAR_I} "\"${ARCANET_GET_VARIANT_NAME}\"")

        string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} ${ARCANET_GET})

        break()
      endif ()

    endforeach ()
  endif ()

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_config_add_dep type dependency)
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" ${type})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} "[]")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} 999 "\"${dependency}\"" )

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

function(arct_config_variation)
  set(options BEFORE AFTER)
  set(oneValueArgs VALUE NEW APPEND)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_variation()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  if (ARGS_AFTER)
    set(ARCANET_BEFORE_AFTER "depend_a")
  else ()
    set(ARCANET_BEFORE_AFTER "depend_b")
  endif ()

  if(ARGS_NEW)
    _arct_config_add_dep("${ARCANET_BEFORE_AFTER}" "${ARGS_NEW}${ARGS_VALUE}")
  elseif (ARGS_APPEND)
    if(NOT ARGS_VALUE)
      message(FATAL_ERROR "No VALUE")
    endif ()
    _arct_config_append_var("${ARCANET_BEFORE_AFTER}" "${ARGS_APPEND}" "${ARGS_VALUE}")
  else ()
    message(FATAL_ERROR "No NEW APPEND")
  endif ()

endfunction()

# ----------------------------------------------------------------------------

function(arct_config_dependency)
  set(options BEFORE AFTER)
  set(oneValueArgs NEW)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_variation()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  if (ARGS_AFTER)
    set(ARCANET_BEFORE_AFTER "depend_a")
  else ()
    set(ARCANET_BEFORE_AFTER "depend_b")
  endif ()

  if(ARGS_NEW)
    _arct_config_add_dep("${ARCANET_BEFORE_AFTER}" "${ARGS_NEW}")
  else ()
    message(FATAL_ERROR "No NEW")
  endif ()
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

macro(_arct_arcane_add_option option_name option_value)

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
endmacro()

# ----------------------------------------------------------------------------

function(arct_arcane_option)
  set(options )
  set(oneValueArgs VALUE NEW)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_arcane_option()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  if(ARGS_NEW AND ARGS_VALUE)
    _arct_arcane_add_option("${ARGS_NEW}" "${ARGS_VALUE}")
  else ()
    message(FATAL_ERROR "No NEW or VALUE")
  endif ()
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

macro(_arct_arccher_add_envvar envvar_name envvar_value)

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
endmacro()

# ----------------------------------------------------------------------------

function(arct_arccher_envvar)
  set(options )
  set(oneValueArgs VALUE NEW)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_arccher_envvar()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  if(ARGS_NEW AND ARGS_VALUE)
    _arct_arccher_add_envvar("${ARGS_NEW}" "${ARGS_VALUE}")
  else ()
    message(FATAL_ERROR "No NEW or VALUE")
  endif ()
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

function(arct_arccher_define_working_dir working_dir)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arct_config_create_or_edit_end()' cannot be called without a call to a 'arct_config_create_or_edit_X()' function.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "working_dir" "\"${working_dir}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
