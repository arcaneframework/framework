
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
# ----------------------------------------------------------------------------

# Characters reserved by the ARCT schema for case, common configuration,
# variation and variant names.
set(_ARCT_RESERVED_NAME_CHARS "[:=~!+]")

# Characters reserved for reserved block names ('~' is allowed because it
# defines alternative reserved blocks).
set(_ARCT_RESERVED_BLOCK_CHARS "[:=!+]")

# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------

# Checks that a name (case, common configuration, variation or variant) does
# not contain any character reserved by the ARCT schema.
function(_arct_check_name name kind)
  string(REGEX MATCH "${_ARCT_RESERVED_NAME_CHARS}" _bad_char "${name}")
  if (_bad_char)
    message(FATAL_ERROR "Name '${name}' (${kind}) contains the reserved character '${_bad_char}'. Characters ':', '=', '~', '!' and '+' cannot be used.")
  endif ()
endfunction()

# ----------------------------------------------------------------------------

# Stores in 'out_var' the list of member names of a JSON object. Comment
# properties ('//...') are ignored, as specified by the ARCT schema.
function(_arct_object_keys json out_var)
  set(_keys "")
  string(JSON _nb ERROR_VARIABLE _err LENGTH "${json}")
  if (_err STREQUAL "NOTFOUND" AND _nb GREATER 0)
    math(EXPR _last "${_nb} - 1")
    foreach (_i RANGE ${_last})
      string(JSON _key ERROR_VARIABLE _err2 MEMBER "${json}" ${_i})
      if (NOT _err2 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error: unable to read a JSON member.")
      endif ()
      if (NOT _key MATCHES "^//")
        list(APPEND _keys "${_key}")
      endif ()
    endforeach ()
  endif ()
  set(${out_var} "${_keys}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

# Checks the reserved block names of a configuration block: every key must be
# '_' (the ARCT reserved block) or a program name. Only '~' is allowed in
# program names (alternative blocks, e.g. 'programme_1~init'). Comment
# properties ('//...') never reach this function: '_arct_object_keys()' has
# already filtered them out.
function(_arct_check_config_block block_json block_desc)
  _arct_object_keys("${block_json}" _keys)
  foreach (_key IN LISTS _keys)
    if (_key STREQUAL "_")
      continue()
    endif ()
    string(REGEX MATCH "${_ARCT_RESERVED_BLOCK_CHARS}" _bad_char "${_key}")
    if (_bad_char)
      message(FATAL_ERROR "Reserved block name '${_key}' (${block_desc}) contains the reserved character '${_bad_char}'. Only '~' (alternative block) is allowed.")
    endif ()
  endforeach ()
endfunction()

# ----------------------------------------------------------------------------

# Validates the global structure of the ARCT file:
# - presence and value of the schema version ('versions._'; this reader
#   supports versions '0' and '1'),
# - type of the root objects ('general', 'cases', 'commons', 'variations'),
# - reserved characters in case, common, variation and variant names,
# - name collisions between 'cases', 'commons' and 'variations',
# - reserved characters in reserved block names.
# The file content is read from the 'ARCANET_JSON' variable of the calling
# scope.
function(_arct_validate_file)
  # Schema version.
  string(JSON _versions ERROR_VARIABLE _err TYPE "${ARCANET_JSON}" "versions")
  if (NOT _err STREQUAL "NOTFOUND")
    message(FATAL_ERROR "The ARCT file must contain a 'versions' object.")
  endif ()
  if (NOT _versions STREQUAL "OBJECT")
    message(FATAL_ERROR "'versions' must be a JSON object.")
  endif ()

  string(JSON _ver ERROR_VARIABLE _err2 GET "${ARCANET_JSON}" "versions" "_")
  if (NOT _err2 STREQUAL "NOTFOUND")
    message(FATAL_ERROR "The ARCT file must contain 'versions._' (the schema version).")
  endif ()

  # The version must be a positive integer or an array of positive integers
  # (the first element is the major version).
  string(JSON _ver_type ERROR_VARIABLE _err3 TYPE "${_ver}")
  set(_major "")
  if (_ver_type STREQUAL "NUMBER")
    if (NOT _ver MATCHES "^[0-9]+$")
      message(FATAL_ERROR "'versions._' must be a positive integer or an array of positive integers (found: '${_ver}').")
    endif ()
    set(_major "${_ver}")
  elseif (_ver_type STREQUAL "ARRAY")
    string(JSON _nb ERROR_VARIABLE _err4 LENGTH "${_ver}")
    if (NOT _err4 STREQUAL "NOTFOUND" OR _nb EQUAL 0)
      message(FATAL_ERROR "'versions._' must be a positive integer or an array of positive integers.")
    endif ()
    string(JSON _first ERROR_VARIABLE _err5 GET "${_ver}" 0)
    string(JSON _first_type ERROR_VARIABLE _err6 TYPE "${_first}")
    if (NOT _first_type STREQUAL "NUMBER" OR NOT _first MATCHES "^[0-9]+$")
      message(FATAL_ERROR "'versions._' must be a positive integer or an array of positive integers.")
    endif ()
    set(_major "${_first}")
  else ()
    message(FATAL_ERROR "'versions._' must be a positive integer or an array of positive integers.")
  endif ()
  if (_major GREATER 1)
    message(FATAL_ERROR "ARCT schema version '${_ver}' is not supported by this reader (supported versions: '0' and '1').")
  endif ()

  # Root objects must be JSON objects when present.
  foreach (_root IN ITEMS general cases commons variations)
    string(JSON _type ERROR_VARIABLE _err7 TYPE "${ARCANET_JSON}" "${_root}")
    if (_err7 STREQUAL "NOTFOUND" AND NOT _type STREQUAL "OBJECT")
      message(FATAL_ERROR "'${_root}' must be a JSON object.")
    endif ()
  endforeach ()

  # Collect the names of cases, commons and variations.
  set(_case_names "")
  set(_common_names "")
  set(_variation_names "")

  string(JSON _obj ERROR_VARIABLE _err8 GET "${ARCANET_JSON}" "cases")
  if (_err8 STREQUAL "NOTFOUND")
    _arct_object_keys("${_obj}" _case_names)
  endif ()
  string(JSON _obj ERROR_VARIABLE _err9 GET "${ARCANET_JSON}" "commons")
  if (_err9 STREQUAL "NOTFOUND")
    _arct_object_keys("${_obj}" _common_names)
  endif ()
  string(JSON _obj ERROR_VARIABLE _err10 GET "${ARCANET_JSON}" "variations")
  if (_err10 STREQUAL "NOTFOUND")
    _arct_object_keys("${_obj}" _variation_names)
  endif ()

  # Names must not use reserved characters and the blocks they describe must
  # be well formed.
  foreach (_name IN LISTS _case_names)
    _arct_check_name("${_name}" "case")
    string(JSON _block ERROR_VARIABLE _err11 GET "${ARCANET_JSON}" "cases" "${_name}")
    _arct_check_config_block("${_block}" "case '${_name}'")
  endforeach ()
  foreach (_name IN LISTS _common_names)
    _arct_check_name("${_name}" "common configuration")
    string(JSON _block ERROR_VARIABLE _err12 GET "${ARCANET_JSON}" "commons" "${_name}")
    _arct_check_config_block("${_block}" "common configuration '${_name}'")
  endforeach ()
  foreach (_name IN LISTS _variation_names)
    _arct_check_name("${_name}" "variation")
    string(JSON _vobj ERROR_VARIABLE _err13 GET "${ARCANET_JSON}" "variations" "${_name}")
    _arct_object_keys("${_vobj}" _variant_names)
    foreach (_variant IN LISTS _variant_names)
      _arct_check_name("${_variant}" "variant")
      string(JSON _block ERROR_VARIABLE _err14 GET "${ARCANET_JSON}" "variations" "${_name}" "${_variant}")
      _arct_check_config_block("${_block}" "variation '${_name}', variant '${_variant}'")
    endforeach ()
  endforeach ()

  string(JSON _obj ERROR_VARIABLE _err15 GET "${ARCANET_JSON}" "general")
  if (_err15 STREQUAL "NOTFOUND")
    _arct_check_config_block("${_obj}" "'general' block")
  endif ()

  # Case, common and variation names must be distinct (JSON allows identical
  # keys in different objects, the ARCT schema does not).
  foreach (_name IN LISTS _case_names)
    if (_name IN_LIST _common_names)
      message(FATAL_ERROR "Name '${_name}' is used both by a case and a common configuration.")
    endif ()
    if (_name IN_LIST _variation_names)
      message(FATAL_ERROR "Name '${_name}' is used both by a case and a variation.")
    endif ()
  endforeach ()
  foreach (_name IN LISTS _common_names)
    if (_name IN_LIST _variation_names)
      message(FATAL_ERROR "Name '${_name}' is used both by a common configuration and a variation.")
    endif ()
  endforeach ()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Parses a 'depend_b'/'depend_a' entry. Result (stored in the parent scope):
#   _ARCT_DEP_MODE    : RESOLVED | ALL | INCLUDE | EXCLUDE
#   _ARCT_DEP_VAR     : variation name
#   _ARCT_DEP_VARIANT : selected variant (RESOLVED only)
#   _ARCT_DEP_FILTERS : variants to include/exclude (INCLUDE/EXCLUDE only)
# Accepted syntaxes: 'variation_1' (all variants are candidates),
# 'variation_1=variant_2' (resolved), 'variation_1!v1!v2' (exclusion filter),
# 'variation_1+v1+v2' (inclusion filter). Symbols cannot be mixed.
function(_arct_parse_dependency dep)
  # At most one kind of symbol is allowed.
  set(_nb_symbols 0)
  foreach (_sym IN ITEMS "=" "+" "!")
    string(FIND "${dep}" "${_sym}" _pos)
    if (NOT _pos EQUAL -1)
      math(EXPR _nb_symbols "${_nb_symbols} + 1")
    endif ()
  endforeach ()
  if (_nb_symbols GREATER 1)
    message(FATAL_ERROR "Symbols '=', '+' and '!' cannot be mixed in dependency '${dep}'.")
  endif ()

  string(FIND "${dep}" "=" _eq)
  if (NOT _eq EQUAL -1)
    # 'variation=variant': the variation is resolved.
    string(SUBSTRING "${dep}" 0 ${_eq} _var)
    math(EXPR _start "${_eq} + 1")
    string(SUBSTRING "${dep}" ${_start} -1 _variant)
    if (NOT _var OR NOT _variant)
      message(FATAL_ERROR "Dependency '${dep}' is malformed.")
    endif ()
    set(_mode "RESOLVED")
  else ()
    string(FIND "${dep}" "+" _plus)
    string(FIND "${dep}" "!" _excl)
    if (_plus EQUAL -1 AND _excl EQUAL -1)
      # No symbol: simple dependency (common configuration or variation
      # without filter).
      set(_var "${dep}")
      set(_filters "")
      set(_mode "ALL")
    else ()
      if (NOT _plus EQUAL -1)
        string(REPLACE "+" ";" _parts "${dep}")
      else ()
        string(REPLACE "!" ";" _parts "${dep}")
      endif ()
      list(GET _parts 0 _var)
      list(REMOVE_AT _parts 0)
      # At least the variation name and one filter are needed.
      if (NOT _var OR NOT _parts)
        message(FATAL_ERROR "Dependency '${dep}' is malformed.")
      endif ()
      set(_filters "${_parts}")
      if (NOT _plus EQUAL -1)
        set(_mode "INCLUDE")
      else ()
        set(_mode "EXCLUDE")
      endif ()
    endif ()
  endif ()

  set(_ARCT_DEP_MODE "${_mode}" PARENT_SCOPE)
  set(_ARCT_DEP_VAR "${_var}" PARENT_SCOPE)
  set(_ARCT_DEP_VARIANT "${_variant}" PARENT_SCOPE)
  set(_ARCT_DEP_FILTERS "${_filters}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Stores 1 in 'out_var' if the variation is already resolved in the case name
# being built, 0 otherwise.
function(_arct_name_has_variation var out_var)
  set(_found 0)
  foreach (_part IN LISTS _ARCT_CTX_NAME)
    string(FIND "${_part}" "=" _eq)
    if (NOT _eq EQUAL -1)
      string(SUBSTRING "${_part}" 0 ${_eq} _part_var)
      if (_part_var STREQUAL "${var}")
        set(_found 1)
        break()
      endif ()
    endif ()
  endforeach ()
  set(${out_var} ${_found} PARENT_SCOPE)
endfunction()

# Stores 1 in 'out_var' if the variation is already resolved by a dependency
# of the current launch path (a 'var=variant' entry has already been read),
# 0 otherwise.
function(_arct_read_has_variation var out_var)
  set(_found 0)
  foreach (_entry IN LISTS _ARCT_CTX_READ)
    string(FIND "${_entry}" "=" _eq)
    if (NOT _eq EQUAL -1)
      string(SUBSTRING "${_entry}" 0 ${_eq} _entry_var)
      if (_entry_var STREQUAL "${var}")
        set(_found 1)
        break()
      endif ()
    endif ()
  endforeach ()
  set(${out_var} ${_found} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Resolution context
# ----------------------------------------------------------------------------

# Marks a configuration block as read. A block can only be read once per use
# case (this also detects dependency cycles). The updated read list is
# exported to the caller scope.
function(_arct_ctx_read_block label)
  list(FIND _ARCT_CTX_READ "${label}" _idx)
  if (NOT _idx EQUAL -1)
    message(FATAL_ERROR "Block '${label}' is read several times or dependencies are cyclic (case '${_ARCT_CTX_BASE}').")
  endif ()
  list(APPEND _ARCT_CTX_READ "${label}")
  set(_ARCT_CTX_READ "${_ARCT_CTX_READ}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

# Exports the mutable resolution context to the caller scope. Expansion and
# enumeration functions each own a scope: context updates must be propagated
# explicitly with PARENT_SCOPE. Calling it several times is harmless.
macro(_arct_ctx_export)
  set(_ARCT_CTX_READ "${_ARCT_CTX_READ}" PARENT_SCOPE)
  set(_ARCT_CTX_PENDING_VARS "${_ARCT_CTX_PENDING_VARS}" PARENT_SCOPE)
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    set("_ARCT_PEND_${_p}" "${_ARCT_PEND_${_p}}" PARENT_SCOPE)
  endforeach ()
endmacro()

# ----------------------------------------------------------------------------

# Saves the mutable context before a backtracking branch. Saved variables are
# suffixed with the nesting depth because branches are recursive.
macro(_arct_ctx_save)
  math(EXPR _ARCT_CTX_DEPTH "${_ARCT_CTX_DEPTH} + 1")
  set("_ARCT_SAVED_PENDING_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_PENDING_VARS}")
  set("_ARCT_SAVED_READ_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_READ}")
  set("_ARCT_SAVED_NAME_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_NAME}")
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    set("_ARCT_SAVED_PEND_${_ARCT_CTX_DEPTH}_${_p}" "${_ARCT_PEND_${_p}}")
  endforeach ()
endmacro()

# Restores the context saved on branch return: pending variations registered
# inside the branch are discarded and the state of the parent branch is
# restored, ready for the next candidate.
macro(_arct_ctx_restore)
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    unset("_ARCT_PEND_${_p}")
  endforeach ()
  set(_ARCT_CTX_PENDING_VARS "${_ARCT_SAVED_PENDING_${_ARCT_CTX_DEPTH}}")
  set(_ARCT_CTX_READ "${_ARCT_SAVED_READ_${_ARCT_CTX_DEPTH}}")
  set(_ARCT_CTX_NAME "${_ARCT_SAVED_NAME_${_ARCT_CTX_DEPTH}}")
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    set("_ARCT_PEND_${_p}" "${_ARCT_SAVED_PEND_${_ARCT_CTX_DEPTH}_${_p}}")
    unset("_ARCT_SAVED_PEND_${_ARCT_CTX_DEPTH}_${_p}")
  endforeach ()
  unset("_ARCT_SAVED_PENDING_${_ARCT_CTX_DEPTH}")
  unset("_ARCT_SAVED_READ_${_ARCT_CTX_DEPTH}")
  unset("_ARCT_SAVED_NAME_${_ARCT_CTX_DEPTH}")
  math(EXPR _ARCT_CTX_DEPTH "${_ARCT_CTX_DEPTH} - 1")
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Expands a dependency list ('depend_b' or 'depend_a') of the ARCT reserved
# block ('_') of a configuration block.
function(_arct_expand_dep_list block_json dep_type)
  string(JSON _deps_type ERROR_VARIABLE _err TYPE "${block_json}" "_" "${dep_type}")
  if (_err STREQUAL "NOTFOUND")
    if (NOT _deps_type STREQUAL "ARRAY")
      message(FATAL_ERROR "Property 'depend_b'/'depend_a' must be a JSON array (case '${_ARCT_CTX_BASE}').")
    endif ()
    string(JSON _nb ERROR_VARIABLE _err2 LENGTH "${block_json}" "_" "${dep_type}")
    if (_err2 STREQUAL "NOTFOUND" AND _nb GREATER 0)
      math(EXPR _last "${_nb} - 1")
      foreach (_i RANGE ${_last})
        string(JSON _dep ERROR_VARIABLE _err3 GET "${block_json}" "_" "${dep_type}" ${_i})
        if (NOT _err3 STREQUAL "NOTFOUND")
          message(FATAL_ERROR "Internal error: unable to read '${dep_type}'.")
        endif ()
        _arct_expand_dependency("${_dep}")
      endforeach ()
    endif ()
  endif ()
  _arct_ctx_export()
endfunction()

# ----------------------------------------------------------------------------

# Recursively expands all the dependencies of a configuration block
# ('depend_b' first, 'depend_a' second). Transitive dependencies of commons
# and variant blocks are followed.
function(_arct_expand_block block_json)
  _arct_expand_dep_list("${block_json}" "depend_b")
  _arct_expand_dep_list("${block_json}" "depend_a")
  _arct_ctx_export()
endfunction()

# ----------------------------------------------------------------------------

# Expands one dependency:
# - RESOLVED ('var=variant'): the variation must exist and must not already
#   be resolved elsewhere in the launch path or in the case name; the variant
#   block is then read recursively;
# - common configuration: the common must exist and its block is read
#   recursively;
# - unresolved variation (optionally filtered with '+'/'!'): the variation is
#   registered as pending. Requesting the same variation twice in one launch
#   path is an error: like a block, a variation can only be requested once
#   (the dependency must be carried by the common ancestor of the requesters).
function(_arct_expand_dependency dep)
  _arct_parse_dependency("${dep}")

  if (_ARCT_DEP_MODE STREQUAL "RESOLVED")
    # A variation cannot be resolved twice (in the name nor by another
    # resolved dependency).
    _arct_name_has_variation("${_ARCT_DEP_VAR}" _in_name)
    if (_in_name)
      message(FATAL_ERROR "Variation '${_ARCT_DEP_VAR}' is resolved several times for case '${_ARCT_CTX_BASE}' (dependency '${dep}').")
    endif ()
    _arct_read_has_variation("${_ARCT_DEP_VAR}" _in_read)
    if (_in_read)
      message(FATAL_ERROR "Variation '${_ARCT_DEP_VAR}' is already resolved by another dependency of case '${_ARCT_CTX_BASE}' (dependency '${dep}').")
    endif ()
    string(JSON _vblock ERROR_VARIABLE _err4 GET "${ARCANET_JSON}" "variations" "${_ARCT_DEP_VAR}" "${_ARCT_DEP_VARIANT}")
    if (NOT _err4 STREQUAL "NOTFOUND")
      message(FATAL_ERROR "Dependency '${dep}' of case '${_ARCT_CTX_BASE}' does not exist: unknown variation or variant.")
    endif ()
    _arct_ctx_read_block("${_ARCT_DEP_VAR}=${_ARCT_DEP_VARIANT}")
    _arct_expand_block("${_vblock}")
  else ()
    # Common configuration: simply read it recursively.
    string(JSON _cblock ERROR_VARIABLE _err5 GET "${ARCANET_JSON}" "commons" "${_ARCT_DEP_VAR}")
    if (_err5 STREQUAL "NOTFOUND")
      _arct_ctx_read_block("${_ARCT_DEP_VAR}")
      _arct_expand_block("${_cblock}")
    else ()
      # Unresolved variation: apply the filters on the available variants and
      # register the variation as pending.
      string(JSON _vobj ERROR_VARIABLE _err6 GET "${ARCANET_JSON}" "variations" "${_ARCT_DEP_VAR}")
      if (NOT _err6 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Dependency '${dep}' of case '${_ARCT_CTX_BASE}' does not exist or is not allowed ('cases' and 'general' blocks cannot be dependencies).")
      endif ()

      _arct_name_has_variation("${_ARCT_DEP_VAR}" _in_name2)
      if (_in_name2)
        message(FATAL_ERROR "Variation '${_ARCT_DEP_VAR}' is resolved several times for case '${_ARCT_CTX_BASE}' (dependency '${dep}').")
      endif ()
      _arct_read_has_variation("${_ARCT_DEP_VAR}" _in_read2)
      if (_in_read2)
        message(FATAL_ERROR "Variation '${_ARCT_DEP_VAR}' is already resolved by another dependency of case '${_ARCT_CTX_BASE}' (dependency '${dep}').")
      endif ()

      _arct_object_keys("${_vobj}" _variants)
      set(_candidates "")
      if (_ARCT_DEP_MODE STREQUAL "INCLUDE")
        # Keep only the requested variants (among the available ones).
        foreach (_v IN LISTS _variants)
          if (_v IN_LIST _ARCT_DEP_FILTERS)
            list(APPEND _candidates "${_v}")
          endif ()
        endforeach ()
      elseif (_ARCT_DEP_MODE STREQUAL "EXCLUDE")
        # Remove the excluded variants.
        foreach (_v IN LISTS _variants)
          if (NOT _v IN_LIST _ARCT_DEP_FILTERS)
            list(APPEND _candidates "${_v}")
          endif ()
        endforeach ()
      else ()
        set(_candidates "${_variants}")
      endif ()

      list(FIND _ARCT_CTX_PENDING_VARS "${_ARCT_DEP_VAR}" _pidx)
      if (NOT _pidx EQUAL -1)
        message(FATAL_ERROR "Variation '${_ARCT_DEP_VAR}' is requested several times in the launch path of case '${_ARCT_CTX_BASE}' (dependency '${dep}'): like a block, an unresolved variation can only be requested once.")
      endif ()
      if (NOT _candidates)
        message(FATAL_ERROR "All variants of variation '${_ARCT_DEP_VAR}' are excluded (case '${_ARCT_CTX_BASE}', dependency '${dep}').")
      endif ()
      list(APPEND _ARCT_CTX_PENDING_VARS "${_ARCT_DEP_VAR}")
      set("_ARCT_PEND_${_ARCT_DEP_VAR}" "${_candidates}")
    endif ()
  endif ()
  _arct_ctx_export()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Enumerates all the complete names of the case. For each pending variation,
# each candidate variant is chosen, its own dependencies are recursively
# expanded (this may register new pending variations), and the enumeration
# continues until no pending variation remains: the complete case name is
# then appended to the results. The context is saved/restored around each
# candidate so that sibling branches do not interfere.
function(_arct_backtrack)
  if (NOT _ARCT_CTX_PENDING_VARS)
    # No pending variation left: the name is complete.
    set(_full_name "${_ARCT_CTX_BASE}")
    if (_ARCT_CTX_NAME)
      string(REPLACE ";" ":" _joined "${_ARCT_CTX_NAME}")
      string(APPEND _full_name ":${_joined}")
    endif ()
    list(APPEND _ARCT_CTX_RESULTS "${_full_name}")
  else ()
    list(GET _ARCT_CTX_PENDING_VARS 0 _var)
    # Build the list of choices ('var=variant') before the loop: local
    # variables do not survive recursive calls.
    set(_choices "")
    foreach (_c IN LISTS _ARCT_PEND_${_var})
      list(APPEND _choices "${_var}=${_c}")
    endforeach ()

    foreach (_choice IN LISTS _choices)
      _arct_ctx_save()
      # The current variation is being resolved: remove it from the pending
      # list (restored on branch return).
      list(REMOVE_AT _ARCT_CTX_PENDING_VARS 0)

      # Split the choice back into variation and variant.
      string(FIND "${_choice}" "=" _eq)
      string(SUBSTRING "${_choice}" 0 ${_eq} _v)
      math(EXPR _start "${_eq} + 1")
      string(SUBSTRING "${_choice}" ${_start} -1 _c)

      # A variation resolved by a dependency cannot reappear in the case
      # name.
      _arct_read_has_variation("${_v}" _already)
      if (_already)
        message(FATAL_ERROR "Variation '${_v}' is already resolved by a dependency of case '${_ARCT_CTX_BASE}' and cannot reappear in the case name.")
      endif ()

      list(APPEND _ARCT_CTX_NAME "${_choice}")
      _arct_ctx_read_block("${_choice}")

      # Expand the dependencies of the chosen variant: they may register new
      # pending variations, which the recursive call below will resolve.
      string(JSON _vblock ERROR_VARIABLE _err GET "${ARCANET_JSON}" "variations" "${_v}" "${_c}")
      if (NOT _err STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Internal error: unable to read variant '${_c}' of variation '${_v}'.")
      endif ()
      _arct_expand_block("${_vblock}")
      _arct_backtrack()

      _arct_ctx_restore()
    endforeach ()
  endif ()

  _arct_ctx_export()
  set(_ARCT_CTX_NAME "${_ARCT_CTX_NAME}" PARENT_SCOPE)
  set(_ARCT_CTX_RESULTS "${_ARCT_CTX_RESULTS}" PARENT_SCOPE)
  set(_ARCT_CTX_DEPTH "${_ARCT_CTX_DEPTH}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Adds one CTest test launching ArcCher on the ARCT file for the given
# (complete) case name. In CMake script mode ('cmake -P') the 'add_test()'
# command is not available: the test is simply skipped (the generated names
# are still reported through 'ARCT_TO_CTEST_TESTS').
function(_arct_add_test)
  set(options)
  set(oneValueArgs ARCCHER_PATH ARCT_CONFIG_PATH ARCT_PATH CASE_NAME CTEST_NAME_PREFIX)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (CMAKE_SCRIPT_MODE_FILE)
    return()
  endif ()

  if (NOT ARGS_ARCT_CONFIG_PATH)
    add_test(NAME "${ARGS_CTEST_NAME_PREFIX}${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}")
  else ()
    add_test(NAME "${ARGS_CTEST_NAME_PREFIX}${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ConfigFile=${ARGS_ARCT_CONFIG_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}" )
  endif ()

endfunction()

# ----------------------------------------------------------------------------
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
