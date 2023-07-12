if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_subdirectory(neo)

option(NEO_BUILD_TEST "Build Neo tests" ON)

if (${NEO_BUILD_TEST})
    find_package(GTest)
    if (${GTEST_FOUND})
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()