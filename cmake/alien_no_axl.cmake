# ----------------------------------------------------------------------------
# Deal with generated files from axl.

function(generateAxl target)

    set(options INSTALL_GENERATED_FILES NO_COPY NO_ARCANE NO_MESH)
    set(oneValueArgs AXL_OPTION_GENERATION_MODE NAMESPACE)
    set(multiValueArgs)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(axl_share_path ${CMAKE_BINARY_DIR}/share/axl)
    set(axl_path ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/axl)

    if (NOT EXISTS ${axl_share_path})
        file(MAKE_DIRECTORY ${axl_share_path})
    endif ()

    set(axl ${ARGS_UNPARSED_ARGUMENTS})

    foreach (axl_file ${axl})

        get_filename_component(name ${axl_file} NAME_WE)
        get_filename_component(directory ${axl_file} DIRECTORY)
        set(axl_src_path ${CMAKE_CURRENT_LIST_DIR}/${directory}/axl)

        target_include_directories(${target} PRIVATE
                $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
                )

        target_link_libraries(${target} PUBLIC strong_options)

        set(axl_files
                ${name}_axl.h
                ${name}_IOptions.h
                ${name}_StrongOptions.h
                )

        set(axl_path_files)
        foreach (faxl ${axl_files})
            configure_file(${axl_src_path}/${faxl} ${axl_path}/${faxl} COPYONLY)
            list(APPEND axl_path_files "${axl_src_path}/${faxl}")
        endforeach ()

        target_sources(${target} PRIVATE ${axl_path_files})

        set_property(GLOBAL APPEND PROPERTY AXL_DB ${file})

        if (ARGS_INSTALL_GENERATED_FILES)
            install(FILES ${axl_path_files} DESTINATION include/${PROJECT_NAME}/axl)
        endif ()

    endforeach ()

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(generateAxlDataBase)

    get_property(axls GLOBAL PROPERTY AXL_DB)

    foreach (axl ${axls})
        set(AXL_STR "${AXL_STR}${axl}\n")
    endforeach ()

    file(WRITE ${PROJECT_BINARY_DIR}/axldb.txt ${AXL_STR})
endfunction()
