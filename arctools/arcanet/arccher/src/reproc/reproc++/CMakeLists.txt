reproc_library(reproc++ CXX)

target_link_libraries(reproc++ PRIVATE
  reproc
  $<$<BOOL:${REPROC_MULTITHREADED}>:Threads::Threads>
)

target_sources(
  reproc++
  PRIVATE src/reproc.cpp
  # We manually propagate reproc's object files until CMake adds support for
  # doing it automatically.
  INTERFACE $<$<BOOL:${REPROC_OBJECT_LIBRARIES}>:$<TARGET_OBJECTS:reproc>>
)
