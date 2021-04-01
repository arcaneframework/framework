add_library(Neo SHARED Utils.h Neo.h Mesh.h Mesh.cpp)

set_target_properties(Neo PROPERTIES PUBLIC_HEADER "Utils.h;Neo.h;Mesh.h")

target_include_directories(Neo INTERFACE
        $<INSTALL_INTERFACE:include>)

install(TARGETS Neo EXPORT NeoConfig PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/neo INCLUDES DESTINATION include)

export (TARGETS Neo FILE ${CMAKE_CURRENT_BINARY_DIR}/NeoConfig.cmake)

install(EXPORT NeoConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)