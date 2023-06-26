add_subdirectory(GTestUnitary)

if (TARGET Alien::alien_semantic_move)
    add_subdirectory(MoveSemantic)
endif ()

if (TARGET Alien::alien_semantic_ref)
    add_subdirectory(RefSemantic)
endif ()
