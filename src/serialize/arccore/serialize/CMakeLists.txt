set(SOURCES
  ISerializer.h
  BasicSerializer.h
  BasicSerializer.cc
  BasicSerializerInternal.h
  SerializeGlobal.h
  SerializeGlobal.cc
)

arccore_add_component_library(serialize
  FILES ${SOURCES}
)

target_link_libraries(arccore_serialize PUBLIC arccore_collections arccore_base)
