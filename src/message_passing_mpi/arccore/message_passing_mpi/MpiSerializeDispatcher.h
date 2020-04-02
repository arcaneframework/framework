// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.h                                    (C) 2000-2020 */
/*                                                                           */
/* Gestion des messages de sérialisation avec MPI.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/Request.h"
#include "arccore/collections/Array.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
class ISerializer;
class BasicSerializer;
class ITraceMng;
}

namespace Arccore::MessagePassing::Mpi
{
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeDispatcher
: public ISerializeDispatcher
{
  friend MpiSerializeMessageList;
  class ReceiveSerializerSubRequest;
  class SendSerializerSubRequest;

 public:

  class SerializeSubRequest
  {
   public:
    static const int MAX_REQUEST_SIZE = 256;
   public:
   public:
    Byte m_bytes[MAX_REQUEST_SIZE];
    Request m_request;
  };

 public:

  MpiSerializeDispatcher(MpiAdapter* adapter);
  ~MpiSerializeDispatcher() override;

 public:

  Ref<ISerializeMessageList> createSerializeMessageListRef() override;

 public:

  // Ces méthodes sont spécifiques à la version MPI.
  //!@{
  Int64 serializeBufferSize() const { return m_serialize_buffer_size; }
  Request sendSerializerWithTag(ISerializer* values,MessageRank rank,MessageTag mpi_tag,bool is_blocking);
  Request recvSerializerBytes(Span<Byte> bytes,MessageRank rank,MessageTag tag,bool is_blocking);
  Request recvSerializerBytes(Span<Byte> bytes,MessageId message_id,bool is_blocking);
  void recvSerializer2(ISerializer* values,MessageRank rank,MessageTag mpi_tag);
  static MessageTag nextSerializeTag(MessageTag tag);
  //!@}

  void broadcastSerializer(ISerializer* values,MessageRank rank);
  Request sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message);
  Request receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message);

  void checkFinishedSubRequests();

  MpiAdapter* adapter() const { return m_adapter; }

 private:

  MpiAdapter* m_adapter = nullptr;
  ITraceMng* m_trace = nullptr;
  Int64 m_serialize_buffer_size;
  Int64 m_max_serialize_buffer_size;
  UniqueArray<SerializeSubRequest*> m_sub_requests;
  bool m_is_trace_serializer= false;
  MPI_Datatype m_byte_serializer_datatype;

 private:

  BasicSerializer* _castSerializer(ISerializer* serializer);
  const BasicSerializer* _castSerializer(const ISerializer* serializer);
  void _checkBigMessage(Int64 message_size);
  Request _sendSerializerBytes(Span<const Byte> bytes,MessageRank rank,
                               MessageTag tag,bool is_blocking);
  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
