include(${BUILD_SYSTEM_PATH}/languages/axl/LoadArcaneAxl.cmake)
include(${BUILD_SYSTEM_PATH}/languages/axl/LoadAxl2ccT4.cmake)
#include(${BUILD_SYSTEM_PATH}/languages/axl/LoadAxl2cc.cmake)
include(${BUILD_SYSTEM_PATH}/languages/axl/LoadAxlCopy.cmake)
include(${BUILD_SYSTEM_PATH}/languages/axl/LoadAxlDoc.cmake)

set(axl_share_path ${CMAKE_BINARY_DIR}/share/axl)

if (NOT EXISTS ${axl_share_path})
    file(MAKE_DIRECTORY ${axl_share_path})
endif ()

function(generateAxl target)

    set(options INSTALL_GENERATED_FILES NO_COPY NO_ARCANE NO_MESH)
    set(oneValueArgs AXL_OPTION_GENERATION_MODE NAMESPACE PROJECT EXPORT)
    set(multiValueArgs)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT ARGS_PROJECT)
        set(axl_path ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/axl)
    else ()
        set(axl_path ${CMAKE_BINARY_DIR}/${ARGS_PROJECT}/axl)
    endif ()

    if (NOT EXISTS ${axl_path})
        file(MAKE_DIRECTORY ${axl_path})
    endif ()

    set(axl ${ARGS_UNPARSED_ARGUMENTS})

    if (VERBOSE)
        set(verbose_args "--verbose=1")
    else ()
        set(verbose_args "--verbose=0")
    endif ()

    foreach (axl_file ${axl})

        get_filename_component(name ${axl_file} NAME_WE)

        if (NOT ARGS_NO_COPY)
            set(copy_args --copy ${axl_share_path}/${name}_${target}.axl)
        endif ()

        set(header_args --header-path ${axl_path})
        set(output_args --output-path ${axl_path})
        target_include_directories(${target} PRIVATE
                $<BUILD_INTERFACE:${axl_path}>
                )

        if (IS_ABSOLUTE ${axl_file})
            set(file ${axl_file})
        else ()
            set(file ${CMAKE_CURRENT_LIST_DIR}/${axl_file})
        endif ()

        if (NOT EXISTS ${file})
            get_source_file_property(is_generated ${file} GENERATED)
            if (NOT ${is_generated})
                logFatalError("axl file ${file} doesn't exist")
            endif ()
        endif ()
        if (AXL2CCT4)
            if (ARGS_NAMESPACE)
                set(namespace ${ARGS_NAMESPACE})
            else ()
                set(namespace "Arcane")
            endif ()
            if (ARGS_NO_ARCANE)
                set(with_arcane no)
            else ()
                set(with_arcane yes)
            endif ()
            if (ARGS_NO_MESH)
                set(with_mesh no)
            else ()
                set(with_mesh yes)
            endif ()
            if (ARGS_EXPORT)
                set(export ${ARGS_EXPORT})
            else ()
                set(export "ARCANE_CORE_EXPORT")
            endif ()

            set(COMMENT_MESSAGE)

            if (ARGS_AXL_OPTION_GENERATION_MODE STREQUAL "ALL")
                set(options_generation_mode all)
                set(generated_files ${axl_path}/${name}_axl.h
                        ${axl_path}/${name}_IOptions.h
                        ${axl_path}/${name}_StrongOptions.h
                        ${axl_path}/${name}_CaseOptionsT.h)
                set(COMMENT_MESSAGE "Building AXL generated file ${name}_axl.h [CaseOptions + StrongOptions]")
            elseif (ARGS_AXL_OPTION_GENERATION_MODE STREQUAL "STRONG_OPTIONS_ONLY")
                set(options_generation_mode strongoption)
                set(generated_files ${axl_path}/${name}_axl.h
                        ${axl_path}/${name}_IOptions.h
                        ${axl_path}/${name}_StrongOptions.h)
                set(COMMENT_MESSAGE "Building AXL generated file ${name}_axl.h [StrongOptions]")
            else ()
                set(options_generation_mode caseoption)
                set(generated_files ${axl_path}/${name}_axl.h)
                set(COMMENT_MESSAGE "Building AXL generated file ${name}_axl.h [CaseOptions]")
            endif ()
            add_custom_command(
                    OUTPUT ${generated_files}
                    DEPENDS ${file} axl
                    COMMAND ${AXL2CC}
                    ARGS ${header_args}
                    ${copy_args}
                    ${output_args}
                    ${verbose_args}
                    --gen-target ${options_generation_mode}
                    --namespace-simple-types ${namespace}
                    --export ${export}
                    --with-arcane ${with_arcane}
                    --with-mesh ${with_mesh}
                    ${file}
                    COMMENT ${COMMENT_MESSAGE}
            )
        else ()
            set(generated_files ${axl_path}/${name}_axl.h)
            add_custom_command(
                    OUTPUT ${generated_files}
                    DEPENDS ${file} axl
                    COMMAND ${AXL2CC}
                    ARGS ${header_args}
                    ${copy_args}
                    ${output_args}
                    ${verbose_args}
                    ${file}
                    COMMENT "Building AXL header ${PROJECT_NAME}/axl/${name}_axl.h"
            )
        endif ()

        foreach (generated_file ${generated_files})
            set_source_files_properties(
                    ${generated_file} PROPERTIES GENERATED ON
            )
        endforeach ()

        target_sources(${target} PRIVATE ${generated_files})

        set_property(GLOBAL APPEND PROPERTY AXL_DB ${file})

        if (ARGS_INSTALL_GENERATED_FILES)
            install(FILES ${generated_files} DESTINATION include/${PROJECT_NAME}/axl)
        endif ()

    endforeach ()

    target_include_directories(${target} PRIVATE
            $<BUILD_INTERFACE:${axl_path}>
            )

endfunction()

function(generateAxlDataBase)

    get_property(axls GLOBAL PROPERTY AXL_DB)

    foreach (axl ${axls})
        set(AXL_STR "${AXL_STR}${axl}\n")
    endforeach ()

    file(WRITE ${PROJECT_BINARY_DIR}/axldb.txt ${AXL_STR})

endfunction()