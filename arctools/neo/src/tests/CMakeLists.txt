add_executable(Neo.tests NeoBaseTest.cpp NeoEvolutiveMeshTest.cpp NeoPolyhedralMeshTest.cpp MeshViewPocTest.cpp NeoMeshAPITest.cpp NeoGraphTest.cpp NeoPerformanceTest.cpp)

target_include_directories(Neo.tests PUBLIC ../.)

find_package(GTest REQUIRED)
target_include_directories(Neo.tests PRIVATE ${GTEST_INCLUDE_DIRS})

find_package(Xdmf QUIET)

if(Xdmf_FOUND)
    target_link_libraries(Neo.tests ${XDMF_LIBNAME} ${XDMF_Xdmf_LIB_DEPENDS})
    target_compile_definitions(Neo.tests PRIVATE HAS_XDMF=1)
    target_include_directories(Neo.tests PUBLIC ${XDMF_INCLUDE_DIRS})
    message("-- XDMF Found ${XDMF_INCLUDE_DIRS}" )
endif(Xdmf_FOUND)

target_link_libraries(Neo.tests GTest::GTest GTest::Main Neo)

file( COPY meshes DESTINATION ${CMAKE_BINARY_DIR}/arctools/neo/src/tests)

include(GNUInstallDirs)

set(_exepath ${CMAKE_BINARY_DIR}/lib)
if (WIN32)
    set_target_properties(Neo.tests
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_exepath}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_exepath}
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_exepath}
            )
else()
    set_target_properties(Neo.tests
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${_exepath}
            )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang on Windows (clang-cl)
    if (WIN32)
        target_compile_options(Neo.tests PRIVATE -EHsc) # need to activate exception on clang-cl from Visual...?
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    target_compile_options(Neo.tests PRIVATE -Wall /bigobj)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    target_compile_options(Neo PRIVATE -Wall)
    if(EXTRA_WARNINGS)
        if(NO_PEDANTIC_ERRORS)
            target_compile_options(Neo PRIVATE -Wextra -Wpedantic -pedantic -Wconversion)
        else()
            target_compile_options(Neo PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wconversion -pedantic-errors)
        endif()
    endif()
endif()

gtest_discover_tests(Neo.tests)

