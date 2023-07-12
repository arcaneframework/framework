add_library(SGraph INTERFACE)

set(header_files ${CMAKE_CURRENT_SOURCE_DIR}/DirectedAcyclicGraph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DirectedGraph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GraphBase.h)

target_sources(SGraph INTERFACE "$<BUILD_INTERFACE:${header_files}>")

target_include_directories(SGraph SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/>)

target_include_directories(SGraph INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    target_compile_options(SGraph INTERFACE -EHsc) # need to activate exception on clang-cl from Visual...?
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    target_compile_options(SGraph INTERFACE -Wall)
    if(EXTRA_WARNINGS)
        if(NO_PEDANTIC_ERRORS)
            target_compile_options(SGraph INTERFACE -Wextra -Wpedantic -pedantic -Wconversion)
        else()
            target_compile_options(SGraph INTERFACE -Wall -Wextra -Wpedantic -pedantic -Wconversion -pedantic-errors)
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++ (not tested)
    target_compile_options(SGraph INTERFACE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    target_compile_options(SGraph INTERFACE -Wall)
endif()

include(GNUInstallDirs)
set_target_properties(SGraph PROPERTIES PUBLIC_HEADER "DirectedAcyclicGraph.h;DirectedGraph.h;GraphBase.h")
install(TARGETS SGraph EXPORT SGraphConfig PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/sgraph)
export (TARGETS SGraph FILE ${CMAKE_BINARY_DIR}/cmake/SGraphConfig.cmake)
install(EXPORT SGraphConfig DESTINATION cmake)