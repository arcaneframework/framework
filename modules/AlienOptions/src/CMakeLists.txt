logStatus(" ** Load library component ${BoldBlue}ALIEN-StrongOptions${ColourReset}")

add_library(ALIEN-StrongOptions INTERFACE)

target_include_directories(ALIEN-StrongOptions INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        )

target_compile_features(ALIEN-StrongOptions INTERFACE cxx_variable_templates)

install(TARGETS ALIEN-StrongOptions
        DESTINATION lib
        EXPORT ${PROJECT_NAME}Targets
        )

install(EXPORT ${PROJECT_NAME}Targets
        DESTINATION lib/cmake
        EXPORT_LINK_INTERFACE_LIBRARIES
        )