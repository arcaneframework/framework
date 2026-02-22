include(CMakePackageConfigHelpers)

set(ENDL "\n")

function(__find_all_targets target targets_list)
    if (NOT TARGET ${target})
        return()
    endif()

    #MESSAGE(STATUS "Targets_list = ${targets_list}")
    get_target_property(LIBRARIES ${target} INTERFACE_LINK_LIBRARIES)
    #MESSAGE(STATUS ">> ${target} depends on ${LIBRARIES}")
    foreach(lib ${LIBRARIES})
        if (TARGET ${lib})
            if(${lib} IN_LIST ${targets_list})
            else()
                list(APPEND ${targets_list} ${lib})
                set(${targets_list} ${${targets_list}} PARENT_SCOPE)
                 __find_all_targets(${lib} ${targets_list})
                 set(${targets_list} ${${targets_list}} PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

function(__check_set_property target property var)
    get_target_property(C_PROP ${target} ${property})
    if(C_PROP)
        set(${var} "set_target_properties(${target} PROPERTIES ${property} \"${C_PROP}\")" PARENT_SCOPE)
    else()
        set(${var} PARENT_SCOPE)
    endif()
endfunction()

function(__append_property target property var)
    __check_set_property(${target} ${property} check_r)
	if (check_r)
	    string(APPEND ${var} "${check_r}" ${ENDL})
	    set(${var} "${${var}}" PARENT_SCOPE)
	 endif()
endfunction()

function(__generate_transitive_dependency target commands)
    set(my_command "if(NOT TARGET ${target})${ENDL}")
	get_target_property(TARGET_TYPE ${target} TYPE)
	if(${TARGET_TYPE} STREQUAL "INTERFACE_LIBRARY")
	    string(APPEND my_command "add_library(${target} INTERFACE IMPORTED)" ${ENDL})
	else()
		string(APPEND my_command "add_library(${target} UNKNOWN IMPORTED)" ${ENDL})
		set(imported_properties IMPORTED_CONFIGURATIONS
                                IMPORTED_IMPLIB
                                IMPORTED_LINK_DEPENDENT_LIBRARIES
                             IMPORTED_LINK_INTERFACE_LANGUAGES
                             IMPORTED_LINK_INTERFACE_LIBRARIES
                             IMPORTED_LINK_INTERFACE_MULTIPLICITY
                             IMPORTED_LOCATION
                             IMPORTED_NO_SONAME
                             IMPORTED
                             IMPORTED_SONAME)
       foreach(config Release Debug RelWithDebInfo)
           string(TOUPPER ${config} CONFIG_)
           list(APPEND imported_properties IMPORTED_LOCATION_${CONFIG_})
           list(APPEND imported_properties IMPORTED_SONAME_${CONFIG_})
       endforeach()
       foreach(property ${imported_properties})
            __append_property(${target} ${property} my_command)
        endforeach()
	endif()
	foreach(property INTERFACE_AUTOUIC_OPTIONS
                     INTERFACE_COMPILE_DEFINITIONS
                     INTERFACE_COMPILE_FEATURES
                     INTERFACE_COMPILE_OPTIONS
                     INTERFACE_INCLUDE_DIRECTORIES
                     INTERFACE_LINK_LIBRARIES
                     INTERFACE_POSITION_INDEPENDENT_CODE
                     INTERFACE_SOURCES
                     INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        __append_property(${target} ${property} my_command)
    endforeach()
	string(APPEND my_command "endif(NOT TARGET ${target})" ${ENDL})
	string(APPEND ${commands} "${my_command}")
	set(${commands} "${${commands}}" PARENT_SCOPE)
endfunction()


macro(GenerateCMakeConfig)
    get_property(EXT_LIBRARIES GLOBAL PROPERTY BUILDSYSTEM_EXTERNAL_LIBRARIES)
    list(REMOVE_DUPLICATES EXT_LIBRARIES)

    SET(LIBS ${EXT_LIBRARIES})

    foreach(target ${EXT_LIBRARIES})
        __find_all_targets(${target} LIBS)
    endforeach()

    #MESSAGE(STATUS "Recursively found ${LIBS}")

    SET(PROJECT_EXTERNAL_LIBRARIES_TARGET "")
    foreach(target ${LIBS})
        __generate_transitive_dependency(${target} PROJECT_EXTERNAL_LIBRARIES_TARGET)
    endforeach()
    #MESSAGE(STATUS "PROJECT_EXTERNAL_LIBRARIES_TARGET = ${PROJECT_EXTERNAL_LIBRARIES_TARGET}")

    set(PROJECT_CONFIG_PATH lib/cmake/${PROJECT_NAME})
    set(PROJECT_DEPENDENCIES_FNAME ${PROJECT_CONFIG_PATH}/${PROJECT_NAME}Dependencies.cmake)
    configure_file(${BUILD_SYSTEM_PATH}/templates/ProjectDependencies.cmake.in
                   ${PROJECT_DEPENDENCIES_FNAME})

    set(PROJECT_TARGET_FNAME ${PROJECT_CONFIG_PATH}/${PROJECT_NAME}Targets.cmake)
    set(PROJECT_CONFIG_FNAME ${PROJECT_CONFIG_PATH}/${PROJECT_NAME}Config.cmake)

    configure_package_config_file(${BUILD_SYSTEM_PATH}/templates/ProjectConfig.cmake.in
      ${PROJECT_CONFIG_FNAME}
      INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
      #PATH_VARS
     NO_CHECK_REQUIRED_COMPONENTS_MACRO )


    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_CONFIG_FNAME}
                  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_DEPENDENCIES_FNAME}
            DESTINATION lib/cmake/${PROJECT_NAME})

endmacro(GenerateCMakeConfig)