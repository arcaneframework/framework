// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Messages.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Identifiant d'un message point à point.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Messages.h"

#include "arccore/base/NotSupportedException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"

#include "arccore/serialize/BasicSerializer.h"
#include "arccore/serialize/internal/BasicSerializerInternal.h"

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/MessageId.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"

/*!
 * \file Messages.h
 *
 * \brief Liste des fonctions d'échange de message.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicSerializeGatherMessage
{
 public:

  void doAllGather(MessagePassing::IMessagePassingMng* pm, const BasicSerializer* send_serializer,
                   BasicSerializer* receive_serializer);

  template <typename DataType> void
  _doGatherOne(MessagePassing::IMessagePassingMng* pm, Span<const DataType> send_values, Span<DataType> recv_buffer)
  {
    UniqueArray<DataType> buf;
    mpAllGatherVariable(pm, send_values, buf);
    recv_buffer.copy(buf);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializeGatherMessage::
doAllGather(MessagePassing::IMessagePassingMng* pm, const BasicSerializer* send_serializer,
            BasicSerializer* receive_serializer)
{
  // TODO:  ne supporte pas encore les types 'Float16', 'BFloat16'
  // 'Float128' et 'Int128' car ces derniers ne sont pas supportés
  // dans les messages MPI.

  const BasicSerializer* sbuf = send_serializer;
  BasicSerializer* recv_buf = receive_serializer;
  BasicSerializer::Impl2* sbuf_p2 = sbuf->m_p2;
  BasicSerializer::Impl2* recv_p2 = recv_buf->m_p2;

  Span<const Real> send_real = sbuf_p2->realBytes();
  Span<const Int16> send_int16 = sbuf_p2->int16Bytes();
  Span<const Int32> send_int32 = sbuf_p2->int32Bytes();
  Span<const Int64> send_int64 = sbuf_p2->int64Bytes();
  Span<const Byte> send_byte = sbuf_p2->byteBytes();
  Span<const Int8> send_int8 = sbuf_p2->int8Bytes();
  Span<const Float16> send_float16 = sbuf_p2->float16Bytes();
  Span<const BFloat16> send_bfloat16 = sbuf_p2->bfloat16Bytes();
  Span<const Float32> send_float32 = sbuf_p2->float32Bytes();
  Span<const Float128> send_float128 = sbuf_p2->float128Bytes();
  Span<const Int128> send_int128 = sbuf_p2->int128Bytes();

  Int64 sizes[11];
  sizes[0] = send_real.size();
  sizes[1] = send_int16.size();
  sizes[2] = send_int32.size();
  sizes[3] = send_int64.size();
  sizes[4] = send_byte.size();
  sizes[5] = send_int8.size();
  sizes[6] = send_float16.size();
  sizes[7] = send_bfloat16.size();
  sizes[8] = send_float32.size();
  sizes[9] = send_float128.size();
  sizes[10] = send_int128.size();

  mpAllReduce(pm, MessagePassing::ReduceSum, ArrayView<Int64>(11, sizes));

  Int64 recv_nb_real = sizes[0];
  Int64 recv_nb_int16 = sizes[1];
  Int64 recv_nb_int32 = sizes[2];
  Int64 recv_nb_int64 = sizes[3];
  Int64 recv_nb_byte = sizes[4];
  Int64 recv_nb_int8 = sizes[5];
  Int64 recv_nb_float16 = sizes[6];
  Int64 recv_nb_bfloat16 = sizes[7];
  Int64 recv_nb_float32 = sizes[8];
  Int64 recv_nb_float128 = sizes[9];
  Int64 recv_nb_int128 = sizes[10];

  if (recv_nb_float16 != 0)
    ARCCORE_THROW(NotImplementedException, "AllGather with serialized type 'float16' is not yet implemented");
  if (recv_nb_bfloat16 != 0)
    ARCCORE_THROW(NotImplementedException, "AllGather with serialized type 'bfloat16' is not yet implemented");
  if (recv_nb_float128 != 0)
    ARCCORE_THROW(NotImplementedException, "AllGather with serialized type 'float128' is not yet implemented");
  if (recv_nb_int128 != 0)
    ARCCORE_THROW(NotImplementedException, "AllGather with serialized type 'int128' is not yet implemented");

  recv_p2->allocateBuffer(recv_nb_real, recv_nb_int16, recv_nb_int32, recv_nb_int64, recv_nb_byte,
                          recv_nb_int8, recv_nb_float16, recv_nb_bfloat16, recv_nb_float32, recv_nb_float128, recv_nb_int128);

  auto recv_p = recv_buf->_p();

  _doGatherOne(pm, send_real, recv_p->getRealBuffer());
  _doGatherOne(pm, send_int32, recv_p->getInt32Buffer());
  _doGatherOne(pm, send_int16, recv_p->getInt16Buffer());
  _doGatherOne(pm, send_int64, recv_p->getInt64Buffer());
  _doGatherOne(pm, send_byte, recv_p->getByteBuffer());
  _doGatherOne(pm, send_int8, recv_p->getInt8Buffer());
  _doGatherOne(pm, send_float32, recv_p->getFloat32Buffer());
}

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une liste de requêtes.
 *
 * \sa IRequestList
 */
Ref<IRequestList>
mpCreateRequestListRef(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->createRequestListRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpWaitAll(IMessagePassingMng* pm, ArrayView<Request> requests)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitAllRequests(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpWait(IMessagePassingMng* pm, Request request)
{
  mpWaitAll(pm, ArrayView<Request>(1, &request));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpWaitSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpTestSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpWait(IMessagePassingMng* pm, ArrayView<Request> requests,
            ArrayView<bool> indexes, eWaitType w_type)
{
  switch (w_type) {
  case WaitAll:
    mpWaitAll(pm, requests);
    indexes.fill(true);
    break;
  case WaitSome:
    mpWaitSome(pm, requests, indexes);
    break;
  case WaitSomeNonBlocking:
    mpTestSome(pm, requests, indexes);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId
mpProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->probe(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo
mpLegacyProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->legacyProbe(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng*
mpSplit(IMessagePassingMng* pm, bool keep)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->commSplit(keep);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpBarrier(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request
mpNonBlockingBarrier(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->nonBlockingBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessageList>
mpCreateSerializeMessageListRef(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->serializeDispatcher();
  return d->createSerializeMessageListRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request
mpSend(IMessagePassingMng* pm, const ISerializer* values,
       const PointToPointMessageInfo& message)
{
  auto d = pm->dispatchers()->serializeDispatcher();
  return d->sendSerializer(values, message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request
mpReceive(IMessagePassingMng* pm, ISerializer* values,
          const PointToPointMessageInfo& message)
{
  auto d = pm->dispatchers()->serializeDispatcher();
  return d->receiveSerializer(values, message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo IControlDispatcher::
legacyProbe(const PointToPointMessageInfo&)
{
  ARCCORE_THROW(NotSupportedException, "pure virtual call to legacyProbe()");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mpAllGather(IMessagePassingMng* pm, const ISerializer* send_serializer, ISerializer* receive_serialize)
{
  auto* s = dynamic_cast<const BasicSerializer*>(send_serializer);
  if (!s)
    ARCCORE_FATAL("send_serializer is not a BasicSerializer");
  auto* r = dynamic_cast<BasicSerializer*>(receive_serialize);
  if (!r)
    ARCCORE_FATAL("receive_serializer is not a BasicSerializer");
  BasicSerializeGatherMessage message;
  message.doAllGather(pm, s, r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage>
mpCreateSerializeMessage(IMessagePassingMng* pm, MessageRank target, ePointToPointMessageType type)
{
  return internal::BasicSerializeMessage::create(MessageRank(pm->commRank()), target, type);
}

Ref<ISerializeMessage>
mpCreateSerializeMessage(IMessagePassingMng* pm, MessageId id)
{
  return internal::BasicSerializeMessage::create(MessageRank(pm->commRank()), id);
}

ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateSendSerializeMessage(IMessagePassingMng* pm, MessageRank destination)
{
  return mpCreateSerializeMessage(pm, destination, ePointToPointMessageType::MsgSend);
}

ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateReceiveSerializeMessage(IMessagePassingMng* pm, MessageRank source)
{
  return mpCreateSerializeMessage(pm, source, ePointToPointMessageType::MsgReceive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  template <typename DataType> inline ITypeDispatcher<DataType>*
  _getDispatcher(IMessagePassingMng* pm)
  {
    ARCCORE_CHECK_POINTER(pm);
    DataType* x = nullptr;
    auto* dispatcher = pm->dispatchers()->dispatcher(x);
    ARCCORE_CHECK_POINTER(dispatcher);
    return dispatcher;
  }
} // namespace

#define ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(type) \
  void mpAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf) \
  { \
    _getDispatcher<type>(pm)->allGather(send_buf, recv_buf); \
  } \
  void mpGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank) \
  { \
    _getDispatcher<type>(pm)->gather(send_buf, recv_buf, rank); \
  } \
  Request mpNonBlockingAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingAllGather(send_buf, recv_buf); \
  } \
  Request mpNonBlockingGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingGather(send_buf, recv_buf, rank); \
  } \
  void mpAllGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf) \
  { \
    _getDispatcher<type>(pm)->allGatherVariable(send_buf, recv_buf); \
  } \
  void mpGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf, Int32 rank) \
  { \
    _getDispatcher<type>(pm)->gatherVariable(send_buf, recv_buf, rank); \
  } \
  Request mpGather(IMessagePassingMng* pm, GatherMessageInfo<type>& gather_info) \
  { \
    return _getDispatcher<type>(pm)->gather(gather_info); \
  } \
  void mpScatterVariable(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 root) \
  { \
    return _getDispatcher<type>(pm)->scatterVariable(send_buf, recv_buf, root); \
  } \
  type mpAllReduce(IMessagePassingMng* pm, eReduceType rt, type v) \
  { \
    return _getDispatcher<type>(pm)->allReduce(rt, v); \
  } \
  void mpAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<type> buf) \
  { \
    _getDispatcher<type>(pm)->allReduce(rt, buf); \
  } \
  Request mpNonBlockingAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<const type> send_buf, Span<type> recv_buf) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingAllReduce(rt, send_buf, recv_buf); \
  } \
  void mpBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank) \
  { \
    _getDispatcher<type>(pm)->broadcast(send_buf, rank); \
  } \
  Request mpNonBlockingBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingBroadcast(send_buf, rank); \
  } \
  void mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank) \
  { \
    _getDispatcher<type>(pm)->send(values, rank, true); \
  } \
  void mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank) \
  { \
    _getDispatcher<type>(pm)->receive(values, rank, true); \
  } \
  Request mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank, bool is_blocked) \
  { \
    return _getDispatcher<type>(pm)->send(values, rank, is_blocked); \
  } \
  Request mpSend(IMessagePassingMng* pm, Span<const type> values, const PointToPointMessageInfo& message) \
  { \
    return _getDispatcher<type>(pm)->send(values, message); \
  } \
  Request mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank, bool is_blocked) \
  { \
    return _getDispatcher<type>(pm)->receive(values, rank, is_blocked); \
  } \
  Request mpReceive(IMessagePassingMng* pm, Span<type> values, const PointToPointMessageInfo& message) \
  { \
    return _getDispatcher<type>(pm)->receive(values, message); \
  } \
  void mpAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count) \
  { \
    return _getDispatcher<type>(pm)->allToAll(send_buf, recv_buf, count); \
  } \
  Request mpNonBlockingAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingAllToAll(send_buf, recv_buf, count); \
  } \
  void mpAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                          ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                          ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) \
  { \
    _getDispatcher<type>(pm)->allToAllVariable(send_buf, send_count, send_index, recv_buf, recv_count, recv_index); \
  } \
  Request mpNonBlockingAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                                        ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                                        ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) \
  { \
    return _getDispatcher<type>(pm)->nonBlockingAllToAllVariable(send_buf, send_count, send_index, recv_buf, recv_count, recv_index); \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(char)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(signed char)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(unsigned char)

ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(short)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(unsigned short)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(int)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(unsigned int)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(long)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(unsigned long)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(long long)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(unsigned long long)

ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(float)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(double)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(long double)

ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(BFloat16)
ARCCORE_GENERATE_MESSAGEPASSING_DEFINITION(Float16)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
