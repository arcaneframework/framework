
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_begin)
  set(options OVERWRITE_ARCT)
  set(oneValueArgs ARCT_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCT_PATH)
    set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
  endif ()

  if (DEFINED ${ARCANET_BEGIN_ON})
    message(FATAL_ERROR "'arcanet_begin()' has been already called. Call 'arcanet_end()' macro to end it.")
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
  set(ARCANET_BEGIN_ON "${ARGS_ARCT_PATH}")
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_end)
  if (NOT DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arcanet_end()' cannot be called without a call to a 'arcanet_begin()' macro.")
  endif ()
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  file(WRITE "${ARCANET_BEGIN_ON}" ${ARCANET_JSON})
  unset(ARCANET_JSON)

  unset(ARCANET_BEGIN_ON)
  unset(ARCANET_JSON_PART_PATH)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_end)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON)
    list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
    list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)
    list(GET ARCANET_JSON_PART_PATH 2 ARCANET_JSON_PART_PATH_2)

    string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2} ${ARCANET_JSON_PART})
  endblock()

  unset(ARCANET_JSON_PART)
  unset(ARCANET_JSON_PART_PATH)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_general)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  set(ARCANET_JSON_PART "")
  set(ARCANET_JSON_PART_PATH "general" "" "")

  block(PROPAGATE ARCANET_JSON ARCANET_JSON_PART)
    list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)

    string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
      set(ARCANET_JSON_PART "{}")
    endif ()
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_common name_common)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  set(ARCANET_JSON_PART "")
  set(ARCANET_JSON_PART_PATH "commons" ${name_common} "")

  block(PROPAGATE ARCANET_JSON ARCANET_JSON_PART)
    list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
    list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)

    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
    endif ()
    string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
      set(ARCANET_JSON_PART "{}")
    endif ()
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_case name_case)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  set(ARCANET_JSON_PART "")
  set(ARCANET_JSON_PART_PATH "cases" ${name_case} "")

  block(PROPAGATE ARCANET_JSON ARCANET_JSON_PART)
    list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
    list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)

    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
    endif ()
    string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
      set(ARCANET_JSON_PART "{}")
    endif ()
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_create_or_edit_variation name_variation name_variant)
  if (DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_X()' has been already called. Call 'arcanet_create_or_edit_end()' macro to end it.")
  endif ()

  set(ARCANET_JSON_PART "")
  set(ARCANET_JSON_PART_PATH "variations" ${name_variation} ${name_variant})

  block(PROPAGATE ARCANET_JSON ARCANET_JSON_PART)
    list(GET ARCANET_JSON_PART_PATH 0 ARCANET_JSON_PART_PATH_0)
    list(GET ARCANET_JSON_PART_PATH 1 ARCANET_JSON_PART_PATH_1)
    list(GET ARCANET_JSON_PART_PATH 2 ARCANET_JSON_PART_PATH_2)

    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} "{}")
    endif ()
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} "{}")
    endif ()
    string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2})
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON SET ${ARCANET_JSON} ${ARCANET_JSON_PART_PATH_0} ${ARCANET_JSON_PART_PATH_1} ${ARCANET_JSON_PART_PATH_2} "{}")
      set(ARCANET_JSON_PART "{}")
    endif ()
  endblock()
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_define_name display_name_case)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "name" "\"${display_name_case}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_add_dependency_a dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
    endif ()
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" "depend_a")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_a" "[]")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_a" 999 "\"${dependency}\"" )
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_add_dependency_b dependency)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_add_dependency()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "{}")
    endif ()
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "_" "depend_b")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_b" "[]")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "_" "depend_b" 999 "\"${dependency}\"" )
  endblock()
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_arcane_set_dataset dataset_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "dataset" "\"${dataset_path}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arcane_add_option option_name option_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "{}")
    endif ()
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arcane" "options")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "options" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arcane" "options" "${option_name}" "\"${option_value}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_mpi_path mpi_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "mpi_binary" "\"${mpi_path}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_add_envvar envvar_name envvar_value)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
    endif ()
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher" "env_var")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "env_var" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "env_var" "${envvar_name}" "\"${envvar_value}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_mpi_nb_procs nb_mpi)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "mpi" "${nb_mpi}")
  endblock()
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_arccher_define_executable exe_path)
  if (NOT DEFINED ARCANET_JSON_PART_PATH)
    message(FATAL_ERROR "'arcanet_create_or_edit_end()' cannot be called without a call to a 'arcanet_create_or_edit_X()' macro.")
  endif ()

  block(PROPAGATE ARCANET_JSON_PART)
    string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "arccher")
    if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
      string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "{}")
    endif ()

    string(JSON ARCANET_JSON_PART SET ${ARCANET_JSON_PART} "arccher" "code_binary" "\"${exe_path}\"")
  endblock()
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(_arcanet_add_test)
  set(options)
  set(oneValueArgs ARCCHER_PATH ARCT_PATH CASE_NAME)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(NAME "${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ParamFile=\"${ARGS_ARCT_PATH}\" -A,Case=\"${ARGS_CASE_NAME}\"")
endmacro()

# ----------------------------------------------------------------------------

macro(_arcanet_add_dep array_dep)

  foreach (VAR_L IN ITEMS ${ARCANET_TESTS_LIST})
    foreach (VAR_M IN ITEMS ${array_dep})
      list(APPEND ARCANET_TEST_LIST_FINAL "${VAR_L}:${VAR_M}")
    endforeach ()
  endforeach ()

  set(ARCANET_TESTS_LIST "${ARCANET_TEST_LIST_FINAL}")
  unset(ARCANET_TEST_LIST_FINAL)
#  message(STATUS "ARCANET_TESTS_LIST=${ARCANET_TESTS_LIST}")

endmacro()

# ----------------------------------------------------------------------------

macro(_arcanet_add_dep_name array_dep)

  foreach (VAR_L IN ITEMS ${ARCANET_TESTS_NAME})
    foreach (VAR_M IN ITEMS ${array_dep})
      list(APPEND ARCANET_TEST_NAME_FINAL "${VAR_L}:${VAR_M}")
    endforeach ()
  endforeach ()

  set(ARCANET_TESTS_NAME "${ARCANET_TEST_NAME_FINAL}")
  unset(ARCANET_TEST_NAME_FINAL)
#  message(STATUS "ARCANET_TESTS_NAME=${ARCANET_TESTS_NAME}")

endmacro()

# ----------------------------------------------------------------------------

macro(_arcanet_check_dep)
#  message(STATUS "ARCANET_DEP=${ARCANET_DEP}")

  string(JSON ARCANET_ARRAY_SIZE LENGTH ${ARCANET_DEP})

#  message(STATUS "ARCANET_ARRAY_SIZE=${ARCANET_ARRAY_SIZE}")

  if(${ARCANET_ARRAY_SIZE} GREATER 0)
    math(EXPR ARCANET_ARRAY_SIZE "${ARCANET_ARRAY_SIZE} - 1")

    foreach (VAR_J RANGE ${ARCANET_ARRAY_SIZE})
#      message(STATUS "VAR_J=${VAR_J}")
      string(JSON ARCANET_ARRAY_PART GET ${ARCANET_DEP} ${VAR_J})

#      message(STATUS "ARCANET_ARRAY_PART=${ARCANET_ARRAY_PART}")


      # S'il y a un égal, alors c'est une dépendance résolue.
      string(FIND ${ARCANET_ARRAY_PART} "=" ARCANET_EQUAL_POS)
      if(NOT ${ARCANET_EQUAL_POS} EQUAL -1)
        _arcanet_add_dep_name("${ARCANET_ARRAY_PART}")
        continue()
      endif ()
      unset(ARCANET_EQUAL_POS)



      # Si la dépendance est dans "commons", alors pas de résolution à faire.
      string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} "commons" ${ARCANET_ARRAY_PART})
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arcanet_add_dep_name("${ARCANET_ARRAY_PART}")
        continue()
      endif ()
      unset(ARCANET_GET)



      # Sinon, on doit ajouter un test par variation

      string(REPLACE "!" ";" ARCANET_VARIATION_EXCL ${ARCANET_ARRAY_PART})
      list(GET ARCANET_VARIATION_EXCL 0 ARCANET_VARIATION)
      list(POP_FRONT ARCANET_VARIATION_EXCL ARCANET_VARIATION_EXCL)

#      message(STATUS "ARCANET_VARIATION=${ARCANET_VARIATION}")
#      message(STATUS "ARCANET_VARIATION_EXCL=${ARCANET_VARIATION_EXCL}")


      string(JSON ARCANET_NB_VARIANTS ERROR_VARIABLE ARCANET_ERROR LENGTH ${ARCANET_JSON} "variations" ${ARCANET_VARIATION})
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        set(ARCANET_NB_VARIANTS 0)
      endif ()

      if(${ARCANET_NB_VARIANTS} GREATER 0)
        math(EXPR ARCANET_NB_VARIANTS "${ARCANET_NB_VARIANTS} - 1")

        foreach (VAR_K RANGE ${ARCANET_NB_VARIANTS})
#          message(STATUS "VAR_K=${VAR_K}")

          string(JSON ARCANET_GET_VARIANT_NAME ERROR_VARIABLE ARCANET_ERROR MEMBER ${ARCANET_JSON} "variations" ${ARCANET_VARIATION} ${VAR_K})
          if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
            message(FATAL_ERROR "Internal error")
          endif ()

#          message(STATUS "ARCANET_GET_VARIANT_NAME=${ARCANET_GET_VARIANT_NAME}")

          if(NOT ${ARCANET_GET_VARIANT_NAME} IN_LIST ARCANET_VARIATION_EXCL)
#            message(STATUS "NOT IN_LIST")
            list(APPEND ARCANET_VARIANT_LIST "${ARCANET_VARIATION}=${ARCANET_GET_VARIANT_NAME}")
          endif ()
        endforeach ()
        _arcanet_add_dep("${ARCANET_VARIANT_LIST}")
        _arcanet_add_dep_name("${ARCANET_VARIANT_LIST}")
        unset(ARCANET_VARIANT_LIST)
      endif ()

    endforeach ()
  endif ()
  unset(ARCANET_ARRAY_SIZE)
  unset(ARCANET_ARRAY_PART)
  unset(ARCANET_ERROR)
  unset(ARCANET_NB_VARIANTS)
  unset(ARCANET_GET_VARIANT_NAME)
  unset(ARCANET_VARIATION_EXCL)
  unset(ARCANET_VARIATION)
endmacro()

# ----------------------------------------------------------------------------

macro(arcanet_to_ctest)
  set(options)
  set(oneValueArgs ARCT_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCT_PATH)
    set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
  endif ()

  if (DEFINED ${ARCANET_BEGIN_ON})
    message(FATAL_ERROR "'arcanet_begin()' has been already called. Call 'arcanet_end()' macro to end it.")
  endif ()
  if (NOT EXISTS "${ARGS_ARCT_PATH}")
    message(FATAL_ERROR "${ARGS_ARCT_PATH} not exist")
  endif ()

  file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)

  string(JSON ARCANET_NB_CASES ERROR_VARIABLE ARCANET_ERROR LENGTH ${ARCANET_JSON} "cases")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    set(ARCANET_NB_CASES 0)
  endif ()

#  message(STATUS "ARCANET_NB_CASES=${ARCANET_NB_CASES}")

  if(${ARCANET_NB_CASES} GREATER 0)
    math(EXPR ARCANET_NB_CASES "${ARCANET_NB_CASES} - 1")

    foreach (VAR_I RANGE ${ARCANET_NB_CASES})
#      message(STATUS "VAR_I=${VAR_I}")

      string(JSON ARCANET_GET_MEMBER ERROR_VARIABLE ARCANET_ERROR MEMBER ${ARCANET_JSON} "cases" ${VAR_I})
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error")
      endif ()

#      message(STATUS "ARCANET_GET_MEMBER=${ARCANET_GET_MEMBER}")

      # Un test sans "arcane":{} doit-il être exécuté ? Oui pour l'instant
      string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} "cases" ${ARCANET_GET_MEMBER} "_")
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        # TODO add_test()
        _arcanet_add_test(ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher" ARCT_PATH "${ARGS_ARCT_PATH}" CASE_NAME "${ARCANET_GET_MEMBER}")
        message(STATUS "add_test(${ARCANET_GET_MEMBER})")
        continue()
      endif ()

      set(ARCANET_TESTS_LIST "${ARCANET_GET_MEMBER}")

      # Noms complets (non utilisée)
      set(ARCANET_TESTS_NAME "${ARCANET_GET_MEMBER}")

#      message(STATUS "ARCANET_JSON_PART=${ARCANET_JSON_PART}")

      string(JSON ARCANET_DEP ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "depend_b")
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arcanet_check_dep()
      endif ()

      string(JSON ARCANET_DEP ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "depend_a")
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arcanet_check_dep()
      endif ()

      foreach (VAR_J IN ITEMS ${ARCANET_TESTS_LIST})
        # TODO add_test()
        _arcanet_add_test(ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher" ARCT_PATH "${ARGS_ARCT_PATH}" CASE_NAME "${VAR_J}")
        message(STATUS "add_test(${VAR_J})")
      endforeach ()

    endforeach ()
  endif ()

  unset(ARCANET_JSON)
  unset(ARCANET_NB_CASES)
  unset(ARCANET_ERROR)
  unset(ARCANET_GET_MEMBER)
  unset(ARCANET_JSON_PART)
  unset(ARCANET_TESTS_LIST)
  unset(ARCANET_TESTS_NAME)
  unset(ARCANET_DEP)

endmacro()

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

arcanet_to_ctest()
#message(FATAL_ERROR "Stop")
