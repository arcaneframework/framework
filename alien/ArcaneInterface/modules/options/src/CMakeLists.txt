logStatus(" ** Load library component ${BoldBlue}alien_strong_options${ColourReset}")

add_library(alien_strong_options INTERFACE)

target_include_directories(alien_strong_options INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        )

target_compile_features(alien_strong_options INTERFACE cxx_variable_templates)

install(TARGETS alien_strong_options
        DESTINATION lib
        EXPORT ${PROJECT_NAME}Targets
        )

install(EXPORT ${PROJECT_NAME}Targets
        DESTINATION lib/cmake/${PROJECT_NAME}
        EXPORT_LINK_INTERFACE_LIBRARIES
        )