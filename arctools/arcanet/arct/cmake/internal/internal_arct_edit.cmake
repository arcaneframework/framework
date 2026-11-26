
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(_arct_internal_begin)
  if (NOT ARGS_ARCT_PATH)
    if(DEFINED ARCT_GLOBAL_ARCT_PATH)
      set(ARGS_ARCT_PATH "${ARCT_GLOBAL_ARCT_PATH}")
    else ()
      set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
    endif ()
  endif ()

  if (DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arct_begin()' has been already called. Call 'arct_end()' function to end it.")
  endif ()

  if (EXISTS "${ARGS_ARCT_PATH}" AND NOT ARGS_OVERWRITE_ARCT)
    file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)
  else ()
    set(ARCANET_JSON "{}")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "{}")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "_" "1")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arcane" "0")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arccher" "0")
  endif ()

  #

  set(ARCANET_BEGIN_ON "${ARGS_ARCT_PATH}" PARENT_SCOPE)
  set(ARCANET_JSON "${ARCANET_JSON}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_end)
  if (NOT DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arct_end()' cannot be called without a call to a 'arct_begin()' function.")
  endif ()

  file(WRITE "${ARCANET_BEGIN_ON}" ${ARCANET_JSON})

  #

  unset(ARCANET_JSON PARENT_SCOPE)
  unset(ARCANET_BEGIN_ON PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(_arct_internal_config_append_var type variation symbol elem_to_add)

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" ${type})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} "[]")
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

        string(FIND ${ARCANET_GET_VARIANT_NAME} "=" ARCANET_POS_SYMBOL)
        if(NOT ARCANET_POS_SYMBOL EQUAL -1)
          message(FATAL_ERROR "Completed variation")
        endif ()

        string(FIND ${ARCANET_GET_VARIANT_NAME} "+" ARCANET_POS_SYMBOL)
        if(NOT ARCANET_POS_SYMBOL EQUAL -1)
          if(NOT ${symbol} STREQUAL "+")
            message(FATAL_ERROR "Mixed symbol")
          endif ()
        endif ()

        string(FIND ${ARCANET_GET_VARIANT_NAME} "!" ARCANET_POS_SYMBOL)
        if(NOT ARCANET_POS_SYMBOL EQUAL -1)
          if(NOT ${symbol} STREQUAL "!")
            message(FATAL_ERROR "Mixed symbol")
          endif ()
        endif ()


        string(APPEND ARCANET_GET_VARIANT_NAME ${elem_to_add})
        string(JSON ARCANET_GET SET ${ARCANET_GET} ${VAR_I} "\"${ARCANET_GET_VARIANT_NAME}\"")

        string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} ${ARCANET_GET})

        set(ARCANET_FOUND 1)
        break()
      endif ()
    endforeach ()

    if(NOT DEFINED ARCANET_FOUND)
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} 999 "\"${variation}${elem_to_add}\"" )
    endif ()
  else()
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" ${type} 999 "\"${variation}${elem_to_add}\"" )
  endif ()


  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_config_add_dep type dependency)
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
# ----------------------------------------------------------------------------

macro(_arct_internal_arcane_add_option option_name option_value)

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
# ----------------------------------------------------------------------------

macro(_arct_internal_arccher_define_mpi_path mpi_path)
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "mpi_binary" "\"${mpi_path}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_arccher_define_mpi_nb_procs nb_mpi)
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "nb_proc" "${nb_mpi}")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_arccher_add_envvar envvar_name envvar_value)

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

macro(_arct_internal_arccher_define_executable exe_path)

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "code_binary" "\"${exe_path}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_arccher_define_working_dir working_dir)

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "working_dir" "\"${working_dir}\"")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------

macro(_arct_internal_arccher_define_nb_thread nb_thread)
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
  endif ()

  string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "nb_thread" "${nb_thread}")

  #

  set(ARCANET_JSON_PART "${ARCANET_JSON_PART}" PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
