add_subdirectory(GTestUnitary)

if (TARGET ALIEN-MoveSemantic)
    add_subdirectory(MoveSemantic)
endif ()

if (TARGET ALIEN-RefSemantic)
    add_subdirectory(RefSemantic)
endif ()
