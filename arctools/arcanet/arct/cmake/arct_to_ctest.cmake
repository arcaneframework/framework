
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Caractères réservés par le schéma ARCT pour les noms de cas, de
# configurations communes, de variations et de variants.
set(_ARCT_RESERVED_NAME_CHARS "[:=~!+]")

# Caractères réservés pour les noms de blocs réservés ('~' est autorisé car
# il permet de définir un bloc réservé alternatif).
set(_ARCT_RESERVED_BLOCK_CHARS "[:=!+]")

# ----------------------------------------------------------------------------

# Vérifie qu'un nom (cas, configuration commune, variation, variant) ne
# contient pas de caractère réservé par le schéma ARCT.
function(_arct_check_name name kind)
  string(REGEX MATCH "${_ARCT_RESERVED_NAME_CHARS}" _bad_char "${name}")
  if (_bad_char)
    message(FATAL_ERROR "Le nom '${name}' (${kind}) contient le caractère réservé '${_bad_char}'. Les caractères ':', '=', '~', '!' et '+' ne peuvent pas être utilisés.")
  endif ()
endfunction()

# ----------------------------------------------------------------------------

# Retourne dans 'out_var' la liste des clés d'un objet JSON.
function(_arct_object_keys json out_var)
  set(_keys "")
  string(JSON _nb ERROR_VARIABLE _err LENGTH "${json}")
  if (_err STREQUAL "NOTFOUND" AND _nb GREATER 0)
    math(EXPR _last "${_nb} - 1")
    foreach (_i RANGE ${_last})
      string(JSON _key ERROR_VARIABLE _err2 MEMBER "${json}" ${_i})
      if (NOT _err2 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Erreur interne : lecture d'un membre JSON.")
      endif ()
      list(APPEND _keys "${_key}")
    endforeach ()
  endif ()
  set(${out_var} "${_keys}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

# Vérifie les noms des blocs réservés d'un bloc de configurations.
function(_arct_check_config_block block_json block_desc)
  _arct_object_keys("${block_json}" _keys)
  foreach (_key IN LISTS _keys)
    if (_key STREQUAL "_")
      continue()
    endif ()
    if (_key MATCHES "^//")
      continue()
    endif ()
    string(REGEX MATCH "${_ARCT_RESERVED_BLOCK_CHARS}" _bad_char "${_key}")
    if (_bad_char)
      message(FATAL_ERROR "Le nom de bloc réservé '${_key}' (${block_desc}) contient le caractère réservé '${_bad_char}'. Seul '~' (bloc alternatif) est autorisé.")
    endif ()
  endforeach ()
endfunction()

# ----------------------------------------------------------------------------

# Valide la structure globale du fichier ARCT : version du schéma, types des
# objets racine, noms (caractères réservés, collisions entre 'cases',
# 'commons' et 'variations') et noms des blocs réservés.
function(_arct_validate_file)
  string(JSON _versions ERROR_VARIABLE _err TYPE "${ARCANET_JSON}" "versions")
  if (NOT _err STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Le fichier ARCT doit contenir un objet 'versions'.")
  endif ()
  if (NOT _versions STREQUAL "OBJECT")
    message(FATAL_ERROR "'versions' doit être un objet JSON.")
  endif ()

  string(JSON _ver ERROR_VARIABLE _err2 GET "${ARCANET_JSON}" "versions" "_")
  if (NOT _err2 STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Le fichier ARCT doit contenir 'versions._' (version du schéma).")
  endif ()

  string(JSON _ver_type ERROR_VARIABLE _err3 TYPE "${_ver}")
  set(_major "")
  if (_ver_type STREQUAL "NUMBER")
    if (NOT _ver MATCHES "^[0-9]+$")
      message(FATAL_ERROR "'versions._' doit être un entier positif ou un tableau d'entiers positifs (valeur trouvée : '${_ver}').")
    endif ()
    set(_major "${_ver}")
  elseif (_ver_type STREQUAL "ARRAY")
    string(JSON _nb ERROR_VARIABLE _err4 LENGTH "${_ver}")
    if (NOT _err4 STREQUAL "NOTFOUND" OR _nb EQUAL 0)
      message(FATAL_ERROR "'versions._' doit être un entier positif ou un tableau d'entiers positifs.")
    endif ()
    string(JSON _first ERROR_VARIABLE _err5 GET "${_ver}" 0)
    string(JSON _first_type ERROR_VARIABLE _err6 TYPE "${_first}")
    if (NOT _first_type STREQUAL "NUMBER" OR NOT _first MATCHES "^[0-9]+$")
      message(FATAL_ERROR "'versions._' doit être un entier positif ou un tableau d'entiers positifs.")
    endif ()
    set(_major "${_first}")
  else ()
    message(FATAL_ERROR "'versions._' doit être un entier positif ou un tableau d'entiers positifs.")
  endif ()
  if (_major GREATER 1)
    message(FATAL_ERROR "La version '${_ver}' du schéma ARCT n'est pas supportée par ce lecteur (versions supportées : '0' et '1').")
  endif ()

  foreach (_root IN ITEMS general cases commons variations)
    string(JSON _type ERROR_VARIABLE _err7 TYPE "${ARCANET_JSON}" "${_root}")
    if (_err7 STREQUAL "NOTFOUND" AND NOT _type STREQUAL "OBJECT")
      message(FATAL_ERROR "'${_root}' doit être un objet JSON.")
    endif ()
  endforeach ()

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

  foreach (_name IN LISTS _case_names)
    _arct_check_name("${_name}" "cas")
    string(JSON _block ERROR_VARIABLE _err11 GET "${ARCANET_JSON}" "cases" "${_name}")
    _arct_check_config_block("${_block}" "cas '${_name}'")
  endforeach ()
  foreach (_name IN LISTS _common_names)
    _arct_check_name("${_name}" "configuration commune")
    string(JSON _block ERROR_VARIABLE _err12 GET "${ARCANET_JSON}" "commons" "${_name}")
    _arct_check_config_block("${_block}" "configuration commune '${_name}'")
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
    _arct_check_config_block("${_obj}" "bloc 'general'")
  endif ()

  foreach (_name IN LISTS _case_names)
    if (_name IN_LIST _common_names)
      message(FATAL_ERROR "Le nom '${_name}' est utilisé à la fois par un cas et une configuration commune.")
    endif ()
    if (_name IN_LIST _variation_names)
      message(FATAL_ERROR "Le nom '${_name}' est utilisé à la fois par un cas et une variation.")
    endif ()
  endforeach ()
  foreach (_name IN LISTS _common_names)
    if (_name IN_LIST _variation_names)
      message(FATAL_ERROR "Le nom '${_name}' est utilisé à la fois par une configuration commune et une variation.")
    endif ()
  endforeach ()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Analyse une entrée de 'depend_b'/'depend_a'. Retourne (PARENT_SCOPE) :
#   _ARCT_DEP_MODE    : RESOLVED | ALL | INCLUDE | EXCLUDE
#   _ARCT_DEP_VAR     : nom de la variation
#   _ARCT_DEP_VARIANT : variant désigné (si RESOLVED)
#   _ARCT_DEP_FILTERS : variants à inclure/exclure (si INCLUDE/EXCLUDE)
function(_arct_parse_dependency dep)
  set(_nb_symbols 0)
  foreach (_sym IN ITEMS "=" "+" "!")
    string(FIND "${dep}" "${_sym}" _pos)
    if (NOT _pos EQUAL -1)
      math(EXPR _nb_symbols "${_nb_symbols} + 1")
    endif ()
  endforeach ()
  if (_nb_symbols GREATER 1)
    message(FATAL_ERROR "Les symboles '=', '+' et '!' ne peuvent pas être mélangés dans la dépendance '${dep}'.")
  endif ()

  string(FIND "${dep}" "=" _eq)
  if (NOT _eq EQUAL -1)
    string(SUBSTRING "${dep}" 0 ${_eq} _var)
    math(EXPR _start "${_eq} + 1")
    string(SUBSTRING "${dep}" ${_start} -1 _variant)
    if (NOT _var OR NOT _variant)
      message(FATAL_ERROR "La dépendance '${dep}' est mal formée.")
    endif ()
    set(_mode "RESOLVED")
  else ()
    string(FIND "${dep}" "+" _plus)
    string(FIND "${dep}" "!" _excl)
    if (_plus EQUAL -1 AND _excl EQUAL -1)
      # Aucun symbole : dépendance simple (configuration commune ou variation
      # sans filtre).
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
      if (NOT _var OR NOT _parts)
        message(FATAL_ERROR "La dépendance '${dep}' est mal formée.")
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

# Indique si la variation est déjà résolue dans le nom du cas en cours de
# construction.
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

# Indique si la variation est déjà résolue par une dépendance du chemin de
# lancement (blocs déjà lus).
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
# ----------------------------------------------------------------------------

# Marque un bloc de configurations comme lu. Un bloc ne peut être lu qu'une
# seule fois par cas d'utilisation : cela détecte aussi les cycles de
# dépendances.
function(_arct_ctx_read_block label)
  list(FIND _ARCT_CTX_READ "${label}" _idx)
  if (NOT _idx EQUAL -1)
    message(FATAL_ERROR "Le bloc '${label}' est lu plusieurs fois ou les dépendances sont cycliques (cas '${_ARCT_CTX_BASE}').")
  endif ()
  list(APPEND _ARCT_CTX_READ "${label}")
  set(_ARCT_CTX_READ "${_ARCT_CTX_READ}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------

# Remonte l'état du contexte vers le scope appelant. Les fonctions d'expansion
# et d'énumération ont chacune leur propre portée : les modifications du
# contexte doivent être propagées explicitement.
macro(_arct_ctx_export)
  set(_ARCT_CTX_READ "${_ARCT_CTX_READ}" PARENT_SCOPE)
  set(_ARCT_CTX_PENDING_VARS "${_ARCT_CTX_PENDING_VARS}" PARENT_SCOPE)
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    set("_ARCT_PEND_${_p}" "${_ARCT_PEND_${_p}}" PARENT_SCOPE)
  endforeach ()
endmacro()

# ----------------------------------------------------------------------------

# Sauvegarde l'état modifiable du contexte (profondeur suffixée car les
# branches du backtracking sont imbriquées).
macro(_arct_ctx_save)
  math(EXPR _ARCT_CTX_DEPTH "${_ARCT_CTX_DEPTH} + 1")
  set("_ARCT_SAVED_PENDING_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_PENDING_VARS}")
  set("_ARCT_SAVED_READ_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_READ}")
  set("_ARCT_SAVED_NAME_${_ARCT_CTX_DEPTH}" "${_ARCT_CTX_NAME}")
  foreach (_p IN LISTS _ARCT_CTX_PENDING_VARS)
    set("_ARCT_SAVED_PEND_${_ARCT_CTX_DEPTH}_${_p}" "${_ARCT_PEND_${_p}}")
  endforeach ()
endmacro()

# Restaure l'état sauvegardé au retour d'une branche du backtracking.
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

# Développe la liste de dépendances 'dep_type' ('depend_b' ou 'depend_a')
# du bloc ARCT réservé ('_') d'un bloc de configurations.
function(_arct_expand_dep_list block_json dep_type)
  string(JSON _deps_type ERROR_VARIABLE _err TYPE "${block_json}" "_" "${dep_type}")
  if (_err STREQUAL "NOTFOUND")
    if (NOT _deps_type STREQUAL "ARRAY")
      message(FATAL_ERROR "La propriété 'depend_b'/'depend_a' doit être un tableau JSON (bloc '${_ARCT_CTX_BASE}').")
    endif ()
    string(JSON _nb ERROR_VARIABLE _err2 LENGTH "${block_json}" "_" "${dep_type}")
    if (_err2 STREQUAL "NOTFOUND" AND _nb GREATER 0)
      math(EXPR _last "${_nb} - 1")
      foreach (_i RANGE ${_last})
        string(JSON _dep ERROR_VARIABLE _err3 GET "${block_json}" "_" "${dep_type}" ${_i})
        if (NOT _err3 STREQUAL "NOTFOUND")
          message(FATAL_ERROR "Erreur interne : lecture de '${dep_type}'.")
        endif ()
        _arct_expand_dependency("${_dep}")
      endforeach ()
    endif ()
  endif ()
  _arct_ctx_export()
endfunction()

# Développe récursivement toutes les dépendances d'un bloc de configurations
# (les dépendances transitives des 'commons' et des variants sont suivies).
function(_arct_expand_block block_json)
  _arct_expand_dep_list("${block_json}" "depend_b")
  _arct_expand_dep_list("${block_json}" "depend_a")
  _arct_ctx_export()
endfunction()

# Développe une dépendance : variation résolue, configuration commune ou
# variation non résolue (avec filtres de variants).
function(_arct_expand_dependency dep)
  _arct_parse_dependency("${dep}")

  if (_ARCT_DEP_MODE STREQUAL "RESOLVED")
    _arct_name_has_variation("${_ARCT_DEP_VAR}" _in_name)
    if (_in_name)
      message(FATAL_ERROR "La variation '${_ARCT_DEP_VAR}' est résolue plusieurs fois pour le cas '${_ARCT_CTX_BASE}' (dépendance '${dep}').")
    endif ()
    _arct_read_has_variation("${_ARCT_DEP_VAR}" _in_read)
    if (_in_read)
      message(FATAL_ERROR "La variation '${_ARCT_DEP_VAR}' est déjà résolue par une autre dépendance du cas '${_ARCT_CTX_BASE}' (dépendance '${dep}').")
    endif ()
    string(JSON _vblock ERROR_VARIABLE _err4 GET "${ARCANET_JSON}" "variations" "${_ARCT_DEP_VAR}" "${_ARCT_DEP_VARIANT}")
    if (NOT _err4 STREQUAL "NOTFOUND")
      message(FATAL_ERROR "La dépendance '${dep}' du cas '${_ARCT_CTX_BASE}' n'existe pas : variation ou variant inconnu.")
    endif ()
    _arct_ctx_read_block("${_ARCT_DEP_VAR}=${_ARCT_DEP_VARIANT}")
    _arct_expand_block("${_vblock}")
  else ()
    string(JSON _cblock ERROR_VARIABLE _err5 GET "${ARCANET_JSON}" "commons" "${_ARCT_DEP_VAR}")
    if (_err5 STREQUAL "NOTFOUND")
      _arct_ctx_read_block("${_ARCT_DEP_VAR}")
      _arct_expand_block("${_cblock}")
    else ()
      string(JSON _vobj ERROR_VARIABLE _err6 GET "${ARCANET_JSON}" "variations" "${_ARCT_DEP_VAR}")
      if (NOT _err6 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "La dépendance '${dep}' du cas '${_ARCT_CTX_BASE}' n'existe pas ou n'est pas autorisée (les blocs 'cases' et 'general' ne peuvent pas être des dépendances).")
      endif ()

      _arct_name_has_variation("${_ARCT_DEP_VAR}" _in_name2)
      if (_in_name2)
        message(FATAL_ERROR "La variation '${_ARCT_DEP_VAR}' est résolue plusieurs fois pour le cas '${_ARCT_CTX_BASE}' (dépendance '${dep}').")
      endif ()
      _arct_read_has_variation("${_ARCT_DEP_VAR}" _in_read2)
      if (_in_read2)
        message(FATAL_ERROR "La variation '${_ARCT_DEP_VAR}' est déjà résolue par une autre dépendance du cas '${_ARCT_CTX_BASE}' (dépendance '${dep}').")
      endif ()

      _arct_object_keys("${_vobj}" _variants)
      set(_candidates "")
      if (_ARCT_DEP_MODE STREQUAL "INCLUDE")
        foreach (_v IN LISTS _variants)
          if (_v IN_LIST _ARCT_DEP_FILTERS)
            list(APPEND _candidates "${_v}")
          endif ()
        endforeach ()
      elseif (_ARCT_DEP_MODE STREQUAL "EXCLUDE")
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
        message(FATAL_ERROR "La variation '${_ARCT_DEP_VAR}' est demandée plusieurs fois dans le chemin de lancement du cas '${_ARCT_CTX_BASE}' (dépendance '${dep}') : tout comme un bloc, une variation non résolue ne peut être demandée qu'une seule fois.")
      endif ()
      if (NOT _candidates)
        message(FATAL_ERROR "Tous les variants de la variation '${_ARCT_DEP_VAR}' sont exclus (cas '${_ARCT_CTX_BASE}', dépendance '${dep}').")
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

# Énumère tous les noms complets du cas : pour chaque variation non résolue,
# un choix de variant est fait puis ses dépendances sont développées
# (récursivement) avant de continuer l'énumération.
function(_arct_backtrack)
  if (NOT _ARCT_CTX_PENDING_VARS)
    set(_full_name "${_ARCT_CTX_BASE}")
    if (_ARCT_CTX_NAME)
      string(REPLACE ";" ":" _joined "${_ARCT_CTX_NAME}")
      string(APPEND _full_name ":${_joined}")
    endif ()
    list(APPEND _ARCT_CTX_RESULTS "${_full_name}")
  else ()
    list(GET _ARCT_CTX_PENDING_VARS 0 _var)
    set(_choices "")
    foreach (_c IN LISTS _ARCT_PEND_${_var})
      list(APPEND _choices "${_var}=${_c}")
    endforeach ()

    foreach (_choice IN LISTS _choices)
      _arct_ctx_save()
      list(REMOVE_AT _ARCT_CTX_PENDING_VARS 0)

      string(FIND "${_choice}" "=" _eq)
      string(SUBSTRING "${_choice}" 0 ${_eq} _v)
      math(EXPR _start "${_eq} + 1")
      string(SUBSTRING "${_choice}" ${_start} -1 _c)

      _arct_read_has_variation("${_v}" _already)
      if (_already)
        message(FATAL_ERROR "La variation '${_v}' est déjà résolue par une dépendance du cas '${_ARCT_CTX_BASE}' et ne peut pas réapparaitre dans le nom du cas.")
      endif ()

      list(APPEND _ARCT_CTX_NAME "${_choice}")
      _arct_ctx_read_block("${_choice}")

      string(JSON _vblock ERROR_VARIABLE _err GET "${ARCANET_JSON}" "variations" "${_v}" "${_c}")
      if (NOT _err STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Erreur interne : lecture du variant '${_c}' de la variation '${_v}'.")
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

function(_arct_add_test)
  set(options)
  set(oneValueArgs ARCCHER_PATH ARCT_CONFIG_PATH ARCT_PATH CASE_NAME CTEST_NAME_PREFIX)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCT_CONFIG_PATH)
    add_test(NAME "${ARGS_CTEST_NAME_PREFIX}${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}")
  else ()
    add_test(NAME "${ARGS_CTEST_NAME_PREFIX}${ARGS_CASE_NAME}" COMMAND "${ARGS_ARCCHER_PATH}" "-A,ConfigFile=${ARGS_ARCT_CONFIG_PATH}" "-A,ParamFile=${ARGS_ARCT_PATH}" "-A,Case=${ARGS_CASE_NAME}" )
  endif ()

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Crée un test CTest par cas d'utilisation du fichier ARCT. Les noms de cas
# sont complets : toutes les variations non résolues (directes ou
# transitives) sont résolues par le produit cartésien de leurs variants.
function(arct_to_ctest)
  set(options)
  set(oneValueArgs ARCT_PATH ARCT_CONFIG_PATH CTEST_NAME_PREFIX ARCCHER_PATH)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_ARCCHER_PATH)
    set(ARGS_ARCCHER_PATH "${CMAKE_BINARY_DIR}/_common/build_all/arcane/arccher/ArcCher")
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

  if (NOT EXISTS "${ARGS_ARCT_PATH}")
    message(FATAL_ERROR "Le fichier ARCT '${ARGS_ARCT_PATH}' n'existe pas.")
  endif ()

  file(READ "${ARGS_ARCT_PATH}" ARCANET_JSON)

  _arct_validate_file()

  string(JSON _nb_cases ERROR_VARIABLE _err LENGTH "${ARCANET_JSON}" "cases")
  if (_err STREQUAL "NOTFOUND" AND _nb_cases GREATER 0)
    math(EXPR _last "${_nb_cases} - 1")
    foreach (_i RANGE ${_last})
      string(JSON _case_name ERROR_VARIABLE _err2 MEMBER "${ARCANET_JSON}" "cases" ${_i})
      if (NOT _err2 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Erreur interne : lecture du nom d'un cas.")
      endif ()

      set(_ARCT_CTX_BASE "${_case_name}")
      set(_ARCT_CTX_READ "${_case_name}")
      set(_ARCT_CTX_PENDING_VARS "")
      set(_ARCT_CTX_NAME "")
      set(_ARCT_CTX_RESULTS "")
      set(_ARCT_CTX_DEPTH 0)

      string(JSON _case_block ERROR_VARIABLE _err3 GET "${ARCANET_JSON}" "cases" "${_case_name}")
      if (NOT _err3 STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Erreur interne : lecture du bloc d'un cas.")
      endif ()
      _arct_expand_block("${_case_block}")

      _arct_backtrack()

      foreach (_full_name IN LISTS _ARCT_CTX_RESULTS)
        _arct_add_test(
          ARCCHER_PATH "${ARGS_ARCCHER_PATH}"
          ARCT_CONFIG_PATH "${ARGS_ARCT_CONFIG_PATH}"
          ARCT_PATH "${ARGS_ARCT_PATH}"
          CASE_NAME "${_full_name}"
          CTEST_NAME_PREFIX "${ARGS_CTEST_NAME_PREFIX}"
        )
        message(STATUS "add_test(${_full_name})")
      endforeach ()

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

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
