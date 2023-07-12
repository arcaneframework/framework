include(FindPackageHandleStandardArgs)

macro(_Sphinx_find_executable _exe)
    string(TOUPPER "${_exe}" _uc)
    find_program(
            SPHINX_${_uc}_EXECUTABLE
            NAMES "sphinx-${_exe}" "sphinx-${_exe}.exe")

    if (SPHINX_${_uc}_EXECUTABLE)
        execute_process(
                COMMAND "${SPHINX_${_uc}_EXECUTABLE}" --version
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _output
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        MESSAGE(STATUS "SPHINX_${_uc}_EXECUTABLE = ${SPHINX_${_uc}_EXECUTABLE} version ${_output}")
        if (_result EQUAL 0 AND _output MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)$")
            set(SPHINX_${_uc}_VERSION "${CMAKE_MATCH_1}")
        endif ()

        add_executable(Sphinx::${_exe} IMPORTED GLOBAL)
        set_target_properties(Sphinx::${_exe} PROPERTIES
                IMPORTED_LOCATION "${SPHINX_${_uc}_EXECUTABLE}")
        set(Sphinx_${_exe}_FOUND TRUE)
    else ()
        set(Sphinx_${_exe}_FOUND FALSE)
    endif ()
    unset(_uc)
endmacro()

#
# Find sphinx-build and sphinx-quickstart.
#
_Sphinx_find_executable(build)
_Sphinx_find_executable(quickstart)

find_package_handle_standard_args(
        Sphinx
        VERSION_VAR SPHINX_VERSION
        REQUIRED_VARS SPHINX_BUILD_EXECUTABLE SPHINX_BUILD_VERSION
)


# Generate a conf.py template file using sphinx-quickstart.
#
# sphinx-quickstart allows for quiet operation and a lot of settings can be
# specified as command line arguments, therefore its not required to parse the
# generated conf.py.
function(_Sphinx_generate_confpy _target _cachedir)
    if (NOT TARGET Sphinx::quickstart)
        message(FATAL_ERROR "sphinx-quickstart is not available, needed by"
                "sphinx_add_docs for target ${_target}")
    endif ()

    if (NOT DEFINED SPHINX_PROJECT)
        set(SPHINX_PROJECT ${PROJECT_NAME})
    endif ()

    if (NOT DEFINED SPHINX_AUTHOR)
        set(SPHINX_AUTHOR "${SPHINX_PROJECT} committers")
    endif ()

    if (NOT DEFINED SPHINX_COPYRIGHT)
        string(TIMESTAMP "%Y, ${SPHINX_AUTHOR}" SPHINX_COPYRIGHT)
    endif ()

    if (NOT DEFINED SPHINX_VERSION)
        set(SPHINX_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
    endif ()

    if (NOT DEFINED SPHINX_RELEASE)
        set(SPHINX_RELEASE "${PROJECT_VERSION}")
    endif ()

    if (NOT DEFINED SPHINX_LANGUAGE)
        set(SPHINX_LANGUAGE "en")
    endif ()

    set(_known_exts autodoc doctest intersphinx todo coverage imgmath mathjax
            ifconfig viewcode githubpages)

    if (DEFINED SPHINX_EXTENSIONS)
        foreach (_ext ${SPHINX_EXTENSIONS})
            set(_is_known_ext FALSE)
            foreach (_known_ext ${_known_exsts})
                if (_ext STREQUAL _known_ext)
                    set(_opts "${opts} --ext-${_ext}")
                    set(_is_known_ext TRUE)
                    break()
                endif ()
            endforeach ()
            if (NOT _is_known_ext)
                if (_exts)
                    set(_exts "${_exts},${_ext}")
                else ()
                    set(_exts "${_ext}")
                endif ()
            endif ()
        endforeach ()
    endif ()

    if (_exts)
        set(_exts "--extensions=${_exts}")
    endif ()

    set(_templatedir "${CMAKE_CURRENT_BINARY_DIR}/${_target}.template")
    file(MAKE_DIRECTORY "${_templatedir}")
    execute_process(
            COMMAND "${SPHINX_QUICKSTART_EXECUTABLE}"
            -q --no-makefile --no-batchfile
            -p "${SPHINX_PROJECT}"
            -a "${SPHINX_AUTHOR}"
            -v "${SPHINX_VERSION}"
            -r "${SPHINX_RELEASE}"
            -l "${SPHINX_LANGUAGE}"
            ${_opts} ${_exts} "${_templatedir}"
            RESULT_VARIABLE _result
            OUTPUT_QUIET)

    if (_result EQUAL 0 AND EXISTS "${_templatedir}/conf.py")
        file(COPY "${_templatedir}/conf.py" DESTINATION "${_cachedir}")
    endif ()

    file(REMOVE_RECURSE "${_templatedir}")

    if (NOT _result EQUAL 0 OR NOT EXISTS "${_cachedir}/conf.py")
        message(FATAL_ERROR "Sphinx configuration file not generated for "
                "target ${_target}")
    endif ()
endfunction()

function(sphinx_add_docs _target)
    set(_opts)
    set(_single_opts BUILDER OUTPUT_DIRECTORY SOURCE_DIRECTORY)
    set(_multi_opts BREATHE_PROJECTS)
    cmake_parse_arguments(_args "${_opts}" "${_single_opts}" "${_multi_opts}" ${ARGN})

    unset(SPHINX_BREATHE_PROJECTS)

    if (NOT _args_BUILDER)
        message(FATAL_ERROR "Sphinx builder not specified for target ${_target}")
    elseif (NOT _args_SOURCE_DIRECTORY)
        message(FATAL_ERROR "Sphinx source directory not specified for target ${_target}")
    else ()
        if (NOT IS_ABSOLUTE "${_args_SOURCE_DIRECTORY}")
            get_filename_component(_sourcedir "${_args_SOURCE_DIRECTORY}" ABSOLUTE)
        else ()
            set(_sourcedir "${_args_SOURCE_DIRECTORY}")
        endif ()
        if (NOT IS_DIRECTORY "${_sourcedir}")
            message(FATAL_ERROR "Sphinx source directory '${_sourcedir}' for"
                    "target ${_target} does not exist")
        endif ()
    endif ()

    set(_builder "${_args_BUILDER}")
    if (_args_OUTPUT_DIRECTORY)
        set(_outputdir "${_args_OUTPUT_DIRECTORY}")
    else ()
        set(_outputdir "${CMAKE_CURRENT_BINARY_DIR}/${_target}")
    endif ()


    if (_args_BREATHE_PROJECTS)
        list(APPEND SPHINX_EXTENSIONS breathe)

        foreach (_doxygen_target ${_args_BREATHE_PROJECTS})
            if (TARGET ${_doxygen_target})
                list(APPEND _depends ${_doxygen_target})

                # Doxygen targets are supported. Verify that a Doxyfile exists.
                get_target_property(_dir ${_doxygen_target} BINARY_DIR)
                set(_doxyfile "${_dir}/Doxyfile.${_doxygen_target}")
                if (NOT EXISTS "${_doxyfile}")
                    message(FATAL_ERROR "Target ${_doxygen_target} is not a Doxygen"
                            "target, needed by sphinx_add_docs for target"
                            "${_target}")
                endif ()

                # Read the Doxyfile, verify XML generation is enabled and retrieve the
                # output directory.
                file(READ "${_doxyfile}" _contents)
                if (NOT _contents MATCHES "GENERATE_XML *= *YES")
                    message(FATAL_ERROR "Doxygen target ${_doxygen_target} does not"
                            "generate XML, needed by sphinx_add_docs for"
                            "target ${_target}")
                elseif (_contents MATCHES "OUTPUT_DIRECTORY *= *([^ ][^\n]*)")
                    string(STRIP "${CMAKE_MATCH_1}" _dir)
                    set(_name "${_doxygen_target}")
                    set(_dir "${_dir}/xml")
                else ()
                    message(FATAL_ERROR "Cannot parse Doxyfile generated by Doxygen"
                            "target ${_doxygen_target}, needed by"
                            "sphinx_add_docs for target ${_target}")
                endif ()
            elseif (_doxygen_target MATCHES "([^: ]+) *: *(.*)")
                set(_name "${CMAKE_MATCH_1}")
                string(STRIP "${CMAKE_MATCH_2}" _dir)
            endif ()

            if (_name AND _dir)
                if (_breathe_projects)
                    set(_breathe_projects "${_breathe_projects}, \"${_name}\": \"${_dir}\"")
                else ()
                    set(_breathe_projects "\"${_name}\": \"${_dir}\"")
                endif ()
                if (NOT _breathe_default_project)
                    set(_breathe_default_project "${_name}")
                endif ()
            endif ()
        endforeach ()
    endif ()

    set(_cachedir "${CMAKE_CURRENT_BINARY_DIR}/${_target}.cache")
    file(MAKE_DIRECTORY "${_cachedir}")
    file(MAKE_DIRECTORY "${_cachedir}/_static")

    _Sphinx_generate_confpy(${_target} "${_cachedir}")

    if (_breathe_projects)
        file(APPEND "${_cachedir}/conf.py"
                "\nbreathe_projects = { ${_breathe_projects} }"
                "\nbreathe_default_project = '${_breathe_default_project}'")
    endif ()

    add_custom_target(
            ${_target}
            COMMAND ${SPHINX_BUILD_EXECUTABLE}
            -b ${_builder}
            -d "${CMAKE_CURRENT_BINARY_DIR}/${_target}.cache/_doctrees"
            -c "${CMAKE_CURRENT_BINARY_DIR}/${_target}.cache"
            "${_sourcedir}"
            "${_outputdir}"
            DEPENDS ${_depends})
endfunction()