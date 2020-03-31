set(SOURCES
  BasicSerializeMessage.h
  BasicSerializeMessage.cc
  MessagePassingGlobal.h
  MessagePassingGlobal.cc
  Dispatchers.h
  Dispatchers.cc
  MessageId.h
  MessageId.cc
  MessageTag.h
  MessageTag.cc
  MessageRank.h
  MessageRank.cc
  PointToPointMessageInfo.h
  PointToPointMessageInfo.cc
  Request.h
  Request.cc
  IRequestList.h
  ISerializeMessage.h
  RequestListBase.h
  RequestListBase.cc
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

target_link_libraries(arccore_message_passing PUBLIC arccore_serialize arccore_collections arccore_base)
