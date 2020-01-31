set(SOURCES
  MessagePassingGlobal.h
  MessagePassingGlobal.cc
  Dispatchers.h
  Dispatchers.cc
  Request.h
  Request.cc
  IStat.h
  IDispatchers.h
  ITypeDispatcher.h
  IControlDispatcher.h
  IMessagePassingMng.h
  MessagePassingMng.h
  MessagePassingMng.cc
  Messages.h
  Stat.h
  Stat.cc
)

arccore_add_component_library(message_passing
  FILES ${SOURCES}
)

target_link_libraries(arccore_message_passing PUBLIC arccore_collections arccore_base)
