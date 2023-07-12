add_subdirectory(sgraph)

option(SGRAPH_BUILD_TEST "Build SGraph tests" ON)

if (${SGRAPH_BUILD_TEST})
    find_package(GTest)
    if (${GTEST_FOUND})
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()