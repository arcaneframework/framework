# NB: par défaut les packages sont cherchés dans un répertoire 'packages' situé à l'endroit
#     du script d'appel de load_packages
macro(loadPackage)
  set(options ESSENTIAL)
  set(oneValueArgs NAME PATH)
  set(multiValueArgs COMPONENTS)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif ()

  if (NOT ARGS_NAME)
    logFatalError("load_package error, name is undefined")
  endif ()

  if (NOT ARGS_PATH)
    get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
    set(path ${SELF_DIR}/packages)
  else ()
    if (IS_ABSOLUTE ${ARGS_PATH})
      set(path ${ARGS_PATH})
    else ()
      get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
      set(path ${SELF_DIR}/${ARGS_PATH})
    endif ()
  endif ()

  if (NOT ARGS_META)
    # We have to find where the findPackage file is defined.
    # Order is: local, arccon, cmake
    # Result is stored in ARCCON_FIND_${ARGS_NAME}_ABSPATH,
    # this name has to depend on package name because find_file uses a cache.
    set(ARCCON_FIND_${ARGS_NAME}_ABSPATH)
    set(FIND_PACK_NAME Find${ARGS_NAME}.cmake)
    find_file(ARCCON_FIND_${ARGS_NAME}_ABSPATH ${FIND_PACK_NAME}
      PATHS ${path} ${ARCCON_PACKAGE_DIRS} ${CMAKE_MODULE_PATH}
      NO_DEFAULT_PATH)
    if (NOT ARCCON_FIND_${ARGS_NAME}_ABSPATH)
      logFatalError("Find${ARGS_NAME}.cmake is not found - check PATH")
      #  else()
      #    logStatus("Found: ${ARGS_NAME} (${FIND_PACK_NAME}) as ${ARCCON_FIND_${ARGS_NAME}_ABSPATH}")
    endif (NOT ARCCON_FIND_${ARGS_NAME}_ABSPATH)
  endif ()

  string(TOLOWER ${ARGS_NAME} target)
  string(TOUPPER ${ARGS_NAME} TARGET)
  set(${ARGS_NAME}_FIND_COMPONENTS ${ARGS_COMPONENTS})

  if (ARGS_ESSENTIAL)
    set(${target}_IS_ESSENTIAL ON)
  endif ()

  if (${target}_IS_DISABLED)
    if (${${target}_IS_ESSENTIAL})
      logFatalError("package ${ARGS_NAME} is essential, can't be disabled")
    endif ()
  else (${target}_IS_DISABLED)
    if (ARGS_META)
      create_meta(NAME ${target})
    else (ARGS_META)
      include(${ARCCON_FIND_${ARGS_NAME}_ABSPATH})
      if (TARGET ${target})
        set(${target}_IS_LOADED ON)
        if (WIN32)
          copyAllDllFromTarget(${target})
        endif (WIN32)
      elseif (${TARGET}_FOUND)
        set(${target}_IS_LOADED ON)
      else (TARGET ${target})
        if (${${target}_IS_ESSENTIAL})
          logFatalError("package ${ARGS_NAME} is essential but not found")
        endif (${${target}_IS_ESSENTIAL})
      endif (TARGET ${target})
    endif (ARGS_META)
  endif (${target}_IS_DISABLED)

  get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)
  list(APPEND TARGETS ${target})
  list(REMOVE_DUPLICATES TARGETS)
  set_property(GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS ${TARGETS})
endmacro()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
