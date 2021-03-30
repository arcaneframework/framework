add_library(Neo.lib SHARED Utils.h Neo.h Mesh.h Mesh.cpp)

install(TARGETS Neo.lib EXPORT NeoConfig)
export (TARGETS Neo.lib FILE ${CMAKE_CURRENT_BINARY_DIR}/NeoConfig.cmake)
install(EXPORT NeoConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)