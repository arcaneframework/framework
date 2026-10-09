
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(_arct_add_test)
  set(options)
  set(oneValueArgs ARCCHER_PATH ARCT_CONFIG_PATH ARCT_PATH CASE_NAME)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCT_CONFIG_PATH)
    add_test(NAME "${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}")
  else ()
    add_test(NAME "${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ConfigFile=${ARGS_ARCT_CONFIG_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}" )
  endif ()

endfunction()

# ----------------------------------------------------------------------------

macro(_arct_add_dep array_dep)

  foreach (VAR_L IN ITEMS ${ARCANET_TESTS_LIST})
    foreach (VAR_M IN ITEMS ${array_dep})
      list(APPEND ARCANET_TEST_LIST_FINAL "${VAR_L}:${VAR_M}")
    endforeach ()
  endforeach ()

  set(ARCANET_TESTS_LIST "${ARCANET_TEST_LIST_FINAL}")
  unset(ARCANET_TEST_LIST_FINAL)
  # message(STATUS "ARCANET_TESTS_LIST=${ARCANET_TESTS_LIST}")

endmacro()

# ----------------------------------------------------------------------------

macro(_arct_add_dep_name array_dep)

  foreach (VAR_L IN ITEMS ${ARCANET_TESTS_NAME})
    foreach (VAR_M IN ITEMS ${array_dep})
      list(APPEND ARCANET_TEST_NAME_FINAL "${VAR_L}:${VAR_M}")
    endforeach ()
  endforeach ()

  set(ARCANET_TESTS_NAME "${ARCANET_TEST_NAME_FINAL}")
  unset(ARCANET_TEST_NAME_FINAL)
  # message(STATUS "ARCANET_TESTS_NAME=${ARCANET_TESTS_NAME}")

endmacro()

# ----------------------------------------------------------------------------

function(_arct_check_dep)
  # message(STATUS "ARCANET_DEP=${ARCANET_DEP}")

  string(JSON ARCANET_ARRAY_SIZE LENGTH ${ARCANET_DEP})

  # message(STATUS "ARCANET_ARRAY_SIZE=${ARCANET_ARRAY_SIZE}")

  if (${ARCANET_ARRAY_SIZE} GREATER 0)
    math(EXPR ARCANET_ARRAY_SIZE "${ARCANET_ARRAY_SIZE} - 1")

    foreach (VAR_J RANGE ${ARCANET_ARRAY_SIZE})
      # message(STATUS "VAR_J=${VAR_J}")
      string(JSON ARCANET_ARRAY_PART GET ${ARCANET_DEP} ${VAR_J})

      # message(STATUS "ARCANET_ARRAY_PART=${ARCANET_ARRAY_PART}")


      # S'il y a un égal, alors c'est une dépendance résolue.
      string(FIND ${ARCANET_ARRAY_PART} "=" ARCANET_EQUAL_POS)
      if (NOT ${ARCANET_EQUAL_POS} EQUAL -1)
        _arct_add_dep_name("${ARCANET_ARRAY_PART}")
        continue()
      endif ()
      unset(ARCANET_EQUAL_POS)


      # Si la dépendance est dans "commons", alors pas de résolution à faire.
      string(JSON ARCANET_GET ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} "commons" ${ARCANET_ARRAY_PART})
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arct_add_dep_name("${ARCANET_ARRAY_PART}")
        continue()
      endif ()
      unset(ARCANET_GET)


      # Sinon, on doit ajouter un test par variation

      set(ARCANET_IS_EXCLUDED_VALUES 1)
      string(REPLACE "!" ";" ARCANET_VARIATION_EXCL ${ARCANET_ARRAY_PART})
      list(GET ARCANET_VARIATION_EXCL 0 ARCANET_VARIATION)
      list(POP_FRONT ARCANET_VARIATION_EXCL ARCANET_VARIATION_EXCL)

      if (NOT ARCANET_VARIATION_EXCL)

        set(ARCANET_IS_EXCLUDED_VALUES 0)
        string(REPLACE "+" ";" ARCANET_VARIATION_EXCL ${ARCANET_ARRAY_PART})
        list(GET ARCANET_VARIATION_EXCL 0 ARCANET_VARIATION)
        list(POP_FRONT ARCANET_VARIATION_EXCL ARCANET_VARIATION_EXCL)

        if (NOT ARCANET_VARIATION_EXCL)
          message(FATAL_ERROR "Variation '${ARCANET_ARRAY_PART}' not valid")
        endif ()
      endif ()

      # message(STATUS "ARCANET_VARIATION=${ARCANET_VARIATION}")
      # message(STATUS "ARCANET_VARIATION_EXCL=${ARCANET_VARIATION_EXCL}")


      string(JSON ARCANET_NB_VARIANTS ERROR_VARIABLE ARCANET_ERROR LENGTH ${ARCANET_JSON} "variations" ${ARCANET_VARIATION})
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        set(ARCANET_NB_VARIANTS 0)
      endif ()

      if (${ARCANET_NB_VARIANTS} GREATER 0)
        math(EXPR ARCANET_NB_VARIANTS "${ARCANET_NB_VARIANTS} - 1")

        foreach (VAR_K RANGE ${ARCANET_NB_VARIANTS})
          # message(STATUS "VAR_K=${VAR_K}")

          string(JSON ARCANET_GET_VARIANT_NAME ERROR_VARIABLE ARCANET_ERROR MEMBER ${ARCANET_JSON} "variations" ${ARCANET_VARIATION} ${VAR_K})
          if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
            message(FATAL_ERROR "Internal error")
          endif ()

          # message(STATUS "ARCANET_GET_VARIANT_NAME=${ARCANET_GET_VARIANT_NAME}")

          if (ARCANET_IS_EXCLUDED_VALUES EQUAL 1 AND NOT ${ARCANET_GET_VARIANT_NAME} IN_LIST ARCANET_VARIATION_EXCL)
            list(APPEND ARCANET_VARIANT_LIST "${ARCANET_VARIATION}=${ARCANET_GET_VARIANT_NAME}")
          elseif (ARCANET_IS_EXCLUDED_VALUES EQUAL 0 AND ${ARCANET_GET_VARIANT_NAME} IN_LIST ARCANET_VARIATION_EXCL)
            list(APPEND ARCANET_VARIANT_LIST "${ARCANET_VARIATION}=${ARCANET_GET_VARIANT_NAME}")
          endif ()
        endforeach ()
        _arct_add_dep("${ARCANET_VARIANT_LIST}")
        _arct_add_dep_name("${ARCANET_VARIANT_LIST}")
        unset(ARCANET_VARIANT_LIST)
      endif ()

    endforeach ()
  endif ()

  set(ARCANET_TESTS_LIST "${ARCANET_TESTS_LIST}" PARENT_SCOPE)
  set(ARCANET_TESTS_NAME "${ARCANET_TESTS_NAME}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

function(arct_to_ctest)
  set(options)
  set(oneValueArgs ARCT_PATH ARCT_CONFIG_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

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
  if (NOT EXISTS "${ARGS_ARCT_PATH}")
    message(FATAL_ERROR "${ARGS_ARCT_PATH} not exist")
  endif ()

  file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)

  string(JSON ARCANET_NB_CASES ERROR_VARIABLE ARCANET_ERROR LENGTH ${ARCANET_JSON} "cases")
  if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
    set(ARCANET_NB_CASES 0)
  endif ()

  # message(STATUS "ARCANET_NB_CASES=${ARCANET_NB_CASES}")

  if (${ARCANET_NB_CASES} GREATER 0)
    math(EXPR ARCANET_NB_CASES "${ARCANET_NB_CASES} - 1")

    foreach (VAR_I RANGE ${ARCANET_NB_CASES})
      # message(STATUS "VAR_I=${VAR_I}")

      string(JSON ARCANET_GET_MEMBER ERROR_VARIABLE ARCANET_ERROR MEMBER ${ARCANET_JSON} "cases" ${VAR_I})
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error")
      endif ()

      # message(STATUS "ARCANET_GET_MEMBER=${ARCANET_GET_MEMBER}")

      # Un test sans "arcane":{} doit-il être exécuté ? Oui pour l'instant
      string(JSON ARCANET_JSON_PART ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON} "cases" ${ARCANET_GET_MEMBER} "_")
      if (NOT ARCANET_ERROR STREQUAL "NOTFOUND")
        _arct_add_test(ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher" ARCT_CONFIG_PATH "${ARGS_ARCT_CONFIG_PATH}" ARCT_PATH "${ARGS_ARCT_PATH}" CASE_NAME "${ARCANET_GET_MEMBER}")
        message(STATUS "add_test(${ARCANET_GET_MEMBER})")
        continue()
      endif ()

      set(ARCANET_TESTS_LIST "${ARCANET_GET_MEMBER}")

      # Noms complets (non utilisée)
      set(ARCANET_TESTS_NAME "${ARCANET_GET_MEMBER}")

      # message(STATUS "ARCANET_JSON_PART=${ARCANET_JSON_PART}")

      string(JSON ARCANET_DEP ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "depend_b")
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arct_check_dep()
      endif ()

      string(JSON ARCANET_DEP ERROR_VARIABLE ARCANET_ERROR GET ${ARCANET_JSON_PART} "depend_a")
      if (ARCANET_ERROR STREQUAL "NOTFOUND")
        _arct_check_dep()
      endif ()

      foreach (VAR_J IN ITEMS ${ARCANET_TESTS_LIST})
        _arct_add_test(ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher" ARCT_CONFIG_PATH "${ARGS_ARCT_CONFIG_PATH}" ARCT_PATH "${ARGS_ARCT_PATH}" CASE_NAME "${VAR_J}")
        message(STATUS "add_test(${VAR_J})")
      endforeach ()

    endforeach ()
  endif ()

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arct_to_ctest()
