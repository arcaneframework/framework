add_executable(Neo.tests NeoBaseTest.cpp NeoEvolutiveMeshTest.cpp NeoPolyhedralMeshTest.cpp MeshView.cpp NeoMeshAPITest.cpp)

target_include_directories(Neo.tests PUBLIC ../.)

find_package(GTest REQUIRED)
target_include_directories(Neo.tests PRIVATE ${GTEST_INCLUDE_DIRS})

find_package(Xdmf REQUIRED)

if(Xdmf_FOUND)
    target_link_libraries(Neo.tests ${XDMF_LIBNAME} ${XDMF_Xdmf_LIB_DEPENDS})
    target_compile_definitions(Neo.tests PRIVATE HAS_XDMF=1)
    #include_directories(${XDMF_INCLUDE_DIRS})
    target_include_directories(Neo.tests PUBLIC ${XDMF_INCLUDE_DIRS})
    message("-- XDMF Found ${XDMF_INCLUDE_DIRS}" )
endif(Xdmf_FOUND)

target_link_libraries(Neo.tests GTest::GTest GTest::Main Neo.lib)

gtest_discover_tests(Neo.tests)

file( COPY meshes DESTINATION ${CMAKE_BINARY_DIR}/src/test)

enable_testing()

