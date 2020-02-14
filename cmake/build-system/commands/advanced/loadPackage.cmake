# NB: par défaut les packages sont cherchés dans un répertoire 'packages' situé à l'endroit
#     du script d'appel de load_packages

macro(loadPackage)

    set(options ESSENTIAL)
    set(oneValueArgs NAME PATH)
    set(multiValueArgs)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # for find_package with Foo_ROOT
    cmake_policy(SET CMP0074 NEW)

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    if (NOT ARGS_NAME)
        logFatalError("load_package error, name is undefined")
    endif ()

    # useless with Arccon
#    if (NOT ARGS_PATH)
#        get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
#        set(path ${SELF_DIR}/packages)
#    else ()
#        if (IS_ABSOLUTE ${ARGS_PATH})
#            set(path ${ARGS_PATH})
#        else ()
#            get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
#            set(path ${SELF_DIR}/${ARGS_PATH})
#        endif ()
#    endif ()
#
#    if (NOT EXISTS ${path}/Find${ARGS_NAME}.cmake AND NOT ARGS_META)
#        logFatalError("Find${ARGS_NAME}.cmake is not found - check PATH")
#    endif ()

    string(TOLOWER ${ARGS_NAME} target)
    string(TOUPPER ${ARGS_NAME} TARGET)

    if (ARGS_ESSENTIAL)
        set(${target}_IS_ESSENTIAL ON)
    endif ()

    if (${target}_IS_DISABLED)
        if (${${target}_IS_ESSENTIAL})
            logFatalError("package ${ARGS_NAME} is essential, can't be disabled")
        endif ()
    else ()
        if (ARGS_META)
            create_meta(NAME ${target})
        else ()
            # Arccon
            # we use find_package now !
            if (ARGS_ESSENTIAL)
                find_package(${ARGS_NAME} REQUIRED)
            else ()
                find_package(${ARGS_NAME} QUIET)
            endif ()
            if(${ARGS_NAME}_FOUND)
                # fix for package Arccon et Axlstar
                if (${ARGS_NAME} STREQUAL "Arccon")
                    set(ARCCON_FOUND ON)
                endif ()
                if (${ARGS_NAME} STREQUAL "Axlstar")
                    set(AXLSTAR_FOUND ON)
                endif ()
                if (${ARGS_NAME} STREQUAL "Arcane")
                    set(ARCANE_FOUND ON)
                endif ()
            endif()
            #include(${path}/Find${ARGS_NAME}.cmake)
            if (TARGET ${target})
                set(${target}_IS_LOADED ON)
                if (WIN32)
                    copyAllDllFromTarget(${target})
                endif ()
            elseif (${TARGET}_FOUND)
                set(${target}_IS_LOADED ON)
            else ()
                if (${${target}_IS_ESSENTIAL})
                    logFatalError("package ${ARGS_NAME} is essential but not found")
                endif ()
            endif ()
        endif ()
    endif ()

    get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)
    list(APPEND TARGETS ${target})
    list(REMOVE_DUPLICATES TARGETS)
    set_property(GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS ${TARGETS})

endmacro()
