add_library(Neo SHARED Utils.h Neo.h Mesh.h Mesh.cpp)
add_library(Neo::Neo ALIAS Neo)

set_target_properties(Neo PROPERTIES PUBLIC_HEADER "Utils.h;Neo.h;Mesh.h")

target_include_directories(Neo SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>)

target_compile_definitions(Neo INTERFACE HAS_NEO)

if(WIN32)
    target_compile_options(Neo PRIVATE -Wall)
else()
    target_compile_options(Neo PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wconversion -pedantic-errors)
endif()

include(GNUInstallDirs)

set(_libpath ${CMAKE_BINARY_DIR}/lib)
if (WIN32)
    set_target_properties(Neo
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${_libpath}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_libpath}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_libpath}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_libpath}
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_libpath}
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_libpath}
            )
else()
    set_target_properties(Neo
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${_libpath}
            RUNTIME_OUTPUT_DIRECTORY ${_libpath}
            )
endif()

install(TARGETS Neo EXPORT NeoConfig PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/neo INCLUDES DESTINATION include)

export (TARGETS Neo FILE ${CMAKE_BINARY_DIR}/cmake/NeoConfig.cmake)

install(EXPORT NeoConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake NAMESPACE Neo)