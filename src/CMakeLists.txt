cmake_minimum_required(VERSION 3.11)
project(Neo)

set(CMAKE_CXX_STANDARD 17)


include_directories(. test)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(Xdmf QUIET)

add_executable(Neo.tests
        test/NeoBaseTest.cpp neo/Neo.h test/NeoEvolutiveMeshTest.cpp test/NeoPolyhedralMeshTest.cpp test/MeshView.cpp)

target_link_libraries(Neo.tests GTest::GTest GTest::Main)

if(Xdmf_FOUND)
    target_link_libraries(Neo.tests ${XDMF_LIBNAME} ${XDMF_Xdmf_LIB_DEPENDS})
    target_compile_definitions(Neo.tests PRIVATE HAS_XDMF=1)
#    target_include_directories(Neo.tests PUBLIC ${XDMF_LIBXML2_INCLUDE_DIR})
    message("-- XDMF Found")
endif(Xdmf_FOUND)

gtest_discover_tests(Neo.tests)

