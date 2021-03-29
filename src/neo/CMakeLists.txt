add_library(Neo.lib SHARED Utils.h Neo.h Mesh.h Mesh.cpp)

install(TARGETS Neo.lib PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/neo)