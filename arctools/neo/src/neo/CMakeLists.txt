add_library(Neo SHARED Utils.h Neo.h Mesh.h Mesh.cpp MeshSchedule.cpp MeshSchedule2.cpp MeshScheduleRemove.cpp MeshKernel.h Items.h Family.h Properties.h Properties.cpp)
add_library(Neo::Neo ALIAS Neo)

set_target_properties(Neo PROPERTIES PUBLIC_HEADER "Utils.h;Neo.h;Mesh.h;Items.h;Properties.h;Family.h")

target_include_directories(Neo SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/>)

target_link_libraries(Neo SGraph)

target_compile_definitions(Neo INTERFACE HAS_NEO)

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

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang on Windows (clang-cl)
    if (WIN32)
        target_compile_options(Neo PRIVATE -EHsc) # need to activate exception on clang-cl from Visual...?
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    target_compile_options(Neo PRIVATE -Wall)
    if(EXTRA_WARNINGS)
        if(NO_PEDANTIC_ERRORS)
            target_compile_options(Neo PRIVATE -Wextra -Wpedantic -pedantic -Wconversion)
        else()
            target_compile_options(Neo PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wconversion -pedantic-errors)
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++ (not tested)
    target_compile_options(Neo PRIVATE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    target_compile_options(Neo PRIVATE  /bigobj)
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

install(EXPORT NeoConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake NAMESPACE Neo::)
