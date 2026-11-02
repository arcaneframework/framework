
# ----------------------------------------------------------------------------
#
# Generates CTest tests from an ARCT (ArcaNet) file.
#
# See 'specifications.md' for the full ARCT v1 specification. This file is a
# reader of the 'cases' part of the schema: for each case it enumerates every
# complete case name (one per combination of unresolved variants) and adds one
# CTest test per name. Tests are run through 'ArcCher', which is the actual
# ARCT consumer at runtime.
#
# The enumeration works in three phases:
#   1. validation of the file (schema version, types, names, collisions...),
#   2. recursive expansion of the dependencies of each case ('depend_b' then
#      'depend_a'): resolved variations ('=') and commons are read recursively
#      (cycles and blocks read twice are detected there), unresolved
#      variations are registered as 'pending',
#   3. backtracking on the pending variations: each candidate variant is
#      chosen, its own dependencies are expanded (this may register new
#      pending variations, including inside variant blocks), and the
#      enumeration goes on until no pending variation remains: the complete
#      case name is then emitted.
#
# Implementation note: expansion and enumeration functions are CMake
# functions, not macros, because macro arguments are substituted textually:
# JSON strings (which contain spaces and quotes, as returned by
# 'string(JSON ...)') cannot safely be passed as macro arguments. Each
# function therefore owns a scope and the mutable resolution context
# ('_ARCT_CTX_*' variables) is propagated to the caller with PARENT_SCOPE
# through '_arct_ctx_export()'.


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(${CMAKE_CURRENT_LIST_DIR}/internal/internal_arct_to_ctest.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Creates one CTest test per use case of the ARCT file. All the case names
# are complete: every unresolved variation (direct or transitive) is resolved
# with the cartesian product of its variants.
# The complete case names of the last call are also stored in the
# 'ARCT_TO_CTEST_TESTS' variable of the caller scope.
# Arguments:
#   ARCT_PATH         : ARCT file path (default: 'ARCT_GLOBAL_ARCT_PATH'
#                       variable or '<binary dir>/testlist.arct'),
#   ARCT_CONFIG_PATH  : optional ArcCher configuration file (default:
#                       'ARCT_GLOBAL_CONFIG_PATH' variable if set),
#   CTEST_NAME_PREFIX : prefix added to the test names,
#   ARCCHER_PATH      : ArcCher executable: path, CMake target name or
#                       generator expression, as understood by
#                       'add_test(COMMAND ...)' (default:
#                       'ARCT_GLOBAL_ARCCHER_PATH' variable if set, else
#                       '<binary dir>/bin/ArcCher').
function(arct_to_ctest)
  set(options)
  set(oneValueArgs ARCT_PATH ARCT_CONFIG_PATH CTEST_NAME_PREFIX ARCCHER_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(ARCT_TO_CTEST_TESTS "")

  # The ArcCher executable can be given as a path, a CMake target name or a
  # generator expression: the value is passed as is to 'add_test()'.
  if (NOT ARGS_ARCCHER_PATH)
    if (DEFINED ARCT_GLOBAL_ARCCHER_PATH)
      set(ARGS_ARCCHER_PATH "${ARCT_GLOBAL_ARCCHER_PATH}")
    else ()
      set(ARGS_ARCCHER_PATH "${CMAKE_BINARY_DIR}/bin/ArcCher")
    endif ()
  endif ()

  if (DEFINED ARCANET_BEGIN_ON)
    message(FATAL_ERROR "'arct_begin()' has been already called. Call 'arct_end()' function to end it.")
  endif ()

  if (NOT ARGS_ARCT_PATH)
    if (DEFINED ARCT_GLOBAL_ARCT_PATH)
      set(ARGS_ARCT_PATH "${ARCT_GLOBAL_ARCT_PATH}")
    else ()
      set(ARGS_ARCT_PATH "${CMAKE_BINARY_DIR}/testlist.arct")
    endif ()
  endif ()

  if (NOT ARGS_ARCT_CONFIG_PATH AND DEFINED ARCT_GLOBAL_CONFIG_PATH)
    set(ARGS_ARCT_CONFIG_PATH "${ARCT_GLOBAL_CONFIG_PATH}")
  endif ()

  if (NOT EXISTS "${ARGS_ARCT_PATH}")
    message(FATAL_ERROR "ARCT file '${ARGS_ARCT_PATH}' does not exist.")
  endif ()

  file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)

  _arct_validate_file()

  # Cases are visited in file order. The number of results depends on the
  # unresolved variations of each case.
  string(JSON _nb_cases ERROR_VARIABLE _err LENGTH "${ARCANET_JSON}" "cases")
  if (_err STREQUAL "NOTFOUND" AND _nb_cases GREATER 0)
    math(EXPR _last "${_nb_cases} - 1")
    foreach (_i RANGE ${_last})
      string(JSON _case_name ERROR_VARIABLE _err2 MEMBER "${ARCANET_JSON}" "cases" ${_i})
      if (NOT _err2 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error: unable to read a case name.")
      endif ()
      # Comment properties ('//...') are not cases.
      if (_case_name MATCHES "^//")
        continue()
      endif ()

      # Resolution context of the case. The case itself is read first, so a
      # dependency cannot point back to it.
      set(_ARCT_CTX_BASE "${_case_name}")
      set(_ARCT_CTX_READ "${_case_name}")
      set(_ARCT_CTX_PENDING_VARS "")
      set(_ARCT_CTX_NAME "")
      set(_ARCT_CTX_RESULTS "")
      set(_ARCT_CTX_DEPTH 0)

      string(JSON _case_block ERROR_VARIABLE _err3 GET "${ARCANET_JSON}" "cases" "${_case_name}")
      if (NOT _err3 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error: unable to read the block of a case.")
      endif ()
      _arct_expand_block("${_case_block}")

      _arct_backtrack()

      foreach (_full_name IN LISTS _ARCT_CTX_RESULTS)
        list(APPEND ARCT_TO_CTEST_TESTS "${_full_name}")
        _arct_add_test(
          ARCCHER_PATH "${ARGS_ARCCHER_PATH}"
          ARCT_CONFIG_PATH "${ARGS_ARCT_CONFIG_PATH}"
          ARCT_PATH "${ARGS_ARCT_PATH}"
          CASE_NAME "${_full_name}"
          CTEST_NAME_PREFIX "${ARGS_CTEST_NAME_PREFIX}"
        )
        #message(STATUS "add_test(${_full_name})")
      endforeach ()

      # Cleanup of the case context before the next one.
      foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
        unset("_ARCT_PEND_${_p}")
      endforeach ()
      unset(_ARCT_CTX_BASE)
      unset(_ARCT_CTX_READ)
      unset(_ARCT_CTX_PENDING_VARS)
      unset(_ARCT_CTX_NAME)
      unset(_ARCT_CTX_RESULTS)
      unset(_ARCT_CTX_DEPTH)
    endforeach ()
  endif ()

  set(ARCT_TO_CTEST_TESTS "${ARCT_TO_CTEST_TESTS}" PARENT_SCOPE)

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
