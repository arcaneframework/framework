configure_file(
        ${PROJECT_SOURCE_DIR}/AlienConfig.h.in
        ${PROJECT_BINARY_DIR}/alien/AlienConfig.h
)

install(
        FILES ${PROJECT_BINARY_DIR}/alien/AlienConfig.h
        DESTINATION include/alien
)
