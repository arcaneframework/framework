set(SOURCES
  BasicSerializeMessage.h
  BasicSerializeMessage.cc
  MessagePassingGlobal.h
  MessagePassingGlobal.cc
  Communicator.h
  Communicator.cc
  Dispatchers.h
  Dispatchers.cc
  GatherMessageInfo.h
  GatherMessageInfo.cc
  MessageId.h
  MessageId.cc
  MessageTag.h
  MessageTag.cc
  MessageRank.h
  MessageRank.cc
  MessageSourceInfo.h
  PointToPointMessageInfo.h
  PointToPointMessageInfo.cc
  PointToPointSerializerMng.h
  PointToPointSerializerMng.cc
  Request.h
  Request.cc
  IRequestList.h
  ISerializeMessage.h
  ISerializeDispatcher.h
  RequestListBase.h
  RequestListBase.cc
  ISerializeMessageList.h
  IStat.h
  IDispatchers.h
  ITypeDispatcher.h
  IControlDispatcher.h
  IProfiler.h
  IMessagePassingMng.h
  MessagePassingMng.h
  MessagePassingMng.cc
  Messages.h
  Messages.cc
  Stat.h
  Stat.cc
  SerializeMessageList.h
  SerializeMessageList.cc
  internal/IMachineMemoryWindowBaseInternal.h
  internal/SubRequestCompletionInfo.h
)

arccore_add_component_library(message_passing
  FILES ${SOURCES}
)

target_link_libraries(arccore_message_passing PUBLIC arccore_serialize arccore_collections arccore_trace arccore_base)
