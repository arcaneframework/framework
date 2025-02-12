// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.h                                    (C) 2000-2025 */
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
#include "arccore/serialize/SerializeGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
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
    Byte m_bytes[MAX_REQUEST_SIZE];
    Request m_request;
  };

 public:

  explicit MpiSerializeDispatcher(MpiAdapter* adapter);
  ~MpiSerializeDispatcher() override;

 public:

  Ref<ISerializeMessageList> createSerializeMessageListRef() override;
  Request sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message) override;
  Request receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message) override;

 public:

  // Ces méthodes sont spécifiques à la version MPI.
  //!@{
  Int64 serializeBufferSize() const { return m_serialize_buffer_size; }
  Request legacySendSerializer(ISerializer* values,const PointToPointMessageInfo& message);
  Request sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message,bool force_one_message);
  void legacyReceiveSerializer(ISerializer* values,MessageRank rank,MessageTag mpi_tag);
  void checkFinishedSubRequests();
  MpiAdapter* adapter() const { return m_adapter; }
  static MessageTag nextSerializeTag(MessageTag tag);
  //!@}

  void broadcastSerializer(ISerializer* values,MessageRank rank);
  ITraceMng* traceMng() const { return m_trace; }

 protected:

  // Ceux deux méthodes sont utilisés aussi par 'MpiSerializeMessageList'
  Request _recvSerializerBytes(Span<Byte> bytes,MessageRank rank,MessageTag tag,bool is_blocking);
  Request _recvSerializerBytes(Span<Byte> bytes,MessageId message_id,bool is_blocking);

 private:

  MpiAdapter* m_adapter = nullptr;
  ITraceMng* m_trace = nullptr;
  Int64 m_serialize_buffer_size;
  Int64 m_max_serialize_buffer_size;
  UniqueArray<SerializeSubRequest*> m_sub_requests;
  bool m_is_trace_serializer = false;
  MPI_Datatype m_byte_serializer_datatype;

 private:

  BasicSerializer* _castSerializer(ISerializer* serializer);
  const BasicSerializer* _castSerializer(const ISerializer* serializer);
  void _checkBigMessage(Int64 message_size);
  Request _sendSerializerWithTag(ISerializer* values,MessageRank rank,
                                 MessageTag mpi_tag,bool is_blocking);
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
