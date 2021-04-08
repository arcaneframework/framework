add_library(Neo SHARED Utils.h Neo.h Mesh.h Mesh.cpp)
add_library(Neo::Neo ALIAS Neo)

set_target_properties(Neo PROPERTIES PUBLIC_HEADER "Utils.h;Neo.h;Mesh.h")

target_include_directories(Neo SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>)

target_compile_definitions(Neo INTERFACE HAS_NEO)

target_compile_options(Neo PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wconversion -Wsign-conversion -pedantic-errors)

install(TARGETS Neo EXPORT NeoConfig PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/neo INCLUDES DESTINATION include)

export (TARGETS Neo FILE ${CMAKE_BINARY_DIR}/cmake/NeoConfig.cmake)

install(EXPORT NeoConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake NAMESPACE Neo)