
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_begin)
  if (DEFINED ${ARCANET_BEGIN_ON})
    message(FATAL_ERROR "'arcanet_begin()' has been already called. Call 'arcanet_end()' macro to end it.")
  endif ()
  #  if (NOT EXISTS "${CMAKE_BINARY_DIR}/testlist.arct")
  set(ARCANET_JSON "{}")
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "{}")
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "_" "0")
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arcane" "0")
  string(JSON ARCANET_JSON SET ${ARCANET_JSON} "versions" "arccher" "0")
  #  else ()
  #    file(READ "${CMAKE_BINARY_DIR}/testlist.arct" ARCANET_JSON)
  #  endif ()
  set(ARCANET_BEGIN_ON 1)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_end)
  if (NOT DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arcanet_end()' cannot be called without a call to a 'arcanet_begin()' macro.")
  endif ()
  if (DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  file(WRITE "${CMAKE_BINARY_DIR}/testlist.arct" ${ARCANET_JSON})
  unset(ARCANET_JSON)

  unset(ARCANET_BEGIN_ON)
  unset(ARCANET_CREATE_OR_EDIT_TYPE)
  unset(ARCANET_CREATE_OR_EDIT_NAME)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_end)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()
  unset(ARCANET_CREATE_OR_EDIT_TYPE)
  unset(ARCANET_CREATE_OR_EDIT_NAME)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_general)
  if (DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()
  set(ARCANET_CREATE_OR_EDIT_TYPE "general")
  set(ARCANET_CREATE_OR_EDIT_NAME "")

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_common name_common)
  if (DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()
  set(ARCANET_CREATE_OR_EDIT_TYPE "commons")
  set(ARCANET_CREATE_OR_EDIT_NAME ${name_common})

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_case name_case)
  if (DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()
  set(ARCANET_CREATE_OR_EDIT_TYPE "cases")
  set(ARCANET_CREATE_OR_EDIT_NAME ${name_case})

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME})
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_define_name display_name_case)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "_")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "_" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "_" "name" "\"${display_name_case}\"")
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_arcane_set_dataset dataset_path)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arcane")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arcane" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arcane" "dataset" "\"${dataset_path}\"")
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arcane_add_option option_name option_value)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arcane")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arcane" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arcane" "options")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arcane" "options" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arcane" "options" "${option_name}" "\"${option_value}\"")
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_mpi_path mpi_path)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "mpi_binary" "\"${mpi_path}\"")
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_add_envvar envvar_name envvar_value)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "{}")
  endif ()
  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arccher" "env_var")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "env_var" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "env_var" "${envvar_name}" "\"${envvar_value}\"")
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_mpi_nb_procs nb_mpi)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "mpi" "${nb_mpi}")
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_executable exe_path)
  if (NOT DEFINED ARCANET_CREATE_OR_EDIT_TYPE)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} "arccher")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "{}")
  endif ()
  unset(ARCANET_GET)
  unset(ARCANET_ERROR)

  string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_CREATE_OR_EDIT_TYPE} ${ARCANET_CREATE_OR_EDIT_NAME} "arccher" "code_binary" "\"${exe_path}\"")
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arcanet_begin()

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

arcanet_create_or_edit_case("mon_test_1")
arcanet_define_name("Mon Test 1")
arcanet_arcane_set_dataset("truc.arc")
arcanet_arcane_add_option("MaxIteration" "3")
arcanet_arcane_add_option("MaxIteration" "4")
arcanet_arccher_define_executable(bin_path)
arcanet_arccher_add_envvar("VARIABLE" "VALUE")
arcanet_create_or_edit_end()

arcanet_end()
