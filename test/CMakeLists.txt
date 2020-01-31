add_executable(test_hypre test_hypre.cpp)

target_link_libraries(test_hypre PUBLIC alien_hypre Alien::ALIEN-IFPEN)