// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Messages.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface for the message exchange manager.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGES_H
#define ARCCORE_MESSAGEPASSING_MESSAGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IDispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Request.h"

#include "arccore/base/RefDeclarations.h"
#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(type) \
  /*! AllGather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf); \
  /*! gather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank); \
  /*! Non-blocking AllGather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf); \
  /*! Non-blocking Gather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank); \
  /*! AllGatherVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf); \
  /*! GatherVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf, Int32 rank); \
  /*! Generic Gather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpGather(IMessagePassingMng* pm, GatherMessageInfo<type>& gather_info); \
  /*! ScatterVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpScatterVariable(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 root); \
  /*! AllReduce */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT type \
  mpAllReduce(IMessagePassingMng* pm, eReduceType rt, type v); \
  /*! AllReduce */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<type> buf); \
  /*! Non-blocking AllReduce */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<const type> send_buf, Span<type> recv_buf); \
  /*! Broadcast */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank); \
  /*! Non-blocking Broadcast */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank, bool is_blocked); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpSend(IMessagePassingMng* pm, Span<const type> values, const PointToPointMessageInfo& message); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank, bool is_blocked); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpReceive(IMessagePassingMng* pm, Span<type> values, const PointToPointMessageInfo& message); \
  /*! AllToAll */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count); \
  /*! Non-blocking AllToAll */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count); \
  /*! AllToAllVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                     ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                     ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index); \
  /*! Non-blocking AllToAllVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                                ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                                ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a list of requests.
 *
 * \sa IRequestList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<IRequestList>
mpCreateRequestListRef(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Blocks until the requests in \a requests are finished.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWaitAll(IMessagePassingMng* pm, ArrayView<Request> requests);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Blocks until the request \a request is finished.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWait(IMessagePassingMng* pm, Request request);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Blocks until at least one of the requests in \a request is finished.
 *
 * In return, the array \a indexes contains the value \a true to indicate
 * that a request is finished.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWaitSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Tests if any of the requests in \a request are finished.
 *
 * In return, the array \a indexes contains the value \a true to indicate
 * that a request is finished.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpTestSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief General function for waiting for request completion.
 *
 * Depending on the value of \a wait_type, calls mpWait(), mpWaitSome(), or
 * mpTestSome().
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWait(IMessagePassingMng* pm, ArrayView<Request> requests,
       ArrayView<bool> indexes, eWaitType wait_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Tests if a message is available.
 *
 * This function checks if a message originating from the (rank,tag) pair
 * is available. \a message must have been initialized with a (rank,tag) pair
 * (message.isRankTag() must be true).
 *
 * Returns an instance of \a MessageId.
 *
 * In non-blocking mode, if no message is available, then
 * MessageId::isValid() is false for the returned instance.
 *
 * The semantics are identical to MPI_Mprobe. The returned message is removed
 * from the message list, and thus a subsequent call to this method with the same
 * parameters will return another message or a null message. If you want behavior
 * identical to MPI_Iprobe()/MPI_Probe(), you must use mpLegacyProbe().
 */
ARCCORE_MESSAGEPASSING_EXPORT MessageId
mpProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Tests if a message is available.
 *
 * This function checks if a message originating from the (rank,tag) pair
 * is available. \a message must have been initialized with a (rank,tag) pair
 * (message.isRankTag() must be true).
 *
 * Returns an instance of \a MessageSourceInfo. In non-blocking mode, if no message
 * is available, then MessageSourceInfo::isValid() is false for
 * the returned instance.
 *
 * The semantics are identical to MPI_Probe. Therefore, it is possible
 * to return the same message if this function is called multiple times.
 * It is also not guaranteed that if you perform an mpReceive() with the instance
 * returned, you will get the same message. For all these reasons, it is preferable
 * to use the mpProbe() function.
 */
ARCCORE_MESSAGEPASSING_EXPORT MessageSourceInfo
mpLegacyProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a new instance of \a IMessagePassingMng.
 *
 * \a keep is true if this rank is present in the new communicator.
 *
 * The returned instance must be destroyed by calling the operator
 * operator delete().
 */
ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng*
mpSplit(IMessagePassingMng* pm, bool keep);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Performs a barrier
 *
 * Blocks until all ranks have reached this call.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpBarrier(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Performs a non-blocking barrier.
 */
ARCCORE_MESSAGEPASSING_EXPORT Request
mpNonBlockingBarrier(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a serialization message list.
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessageList>
mpCreateSerializeMessageListRef(IMessagePassingMng* pm);

//! Send message using an ISerializer.
ARCCORE_MESSAGEPASSING_EXPORT Request
mpSend(IMessagePassingMng* pm, const ISerializer* values, const PointToPointMessageInfo& message);

//! Receive message using an ISerializer.
ARCCORE_MESSAGEPASSING_EXPORT Request
mpReceive(IMessagePassingMng* pm, ISerializer* values, const PointToPointMessageInfo& message);

//! allGather() message for serialization
ARCCORE_MESSAGEPASSING_EXPORT void
mpAllGather(IMessagePassingMng* pm, const ISerializer* send_serializer, ISerializer* recv_serializer);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a serialization message.
 *
 * \a type is the message type and \a target is the target.
 * If the message is a send, \a target is the message destination.
 * If the message is a receive, \a target is the message source.
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateSerializeMessage(IMessagePassingMng* pm, MessageRank target, ePointToPointMessageType type);

/*!
 * \brief Creates a serialization message corresponding to \a id.
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateSerializeMessage(IMessagePassingMng* pm, MessageId id);

/*!
 * \brief Creates a serialization message for sending.
 *
 * This method is equivalent to
 * mpCreateSerializeMessage(pm, destination, ePointToPointMessageType::MsgSend).
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateSendSerializeMessage(IMessagePassingMng* pm, MessageRank destination);

/*!
 * \brief Creates a serialization message for receiving.
 *
 * This method is equivalent to
 * mpCreateSerializeMessage(pm, source, ePointToPointMessageType::MsgReceive).
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessage>
mpCreateReceiveSerializeMessage(IMessagePassingMng* pm, MessageRank source);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(signed char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned char)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long long)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(float)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(double)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long double)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(BFloat16)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(Float16)

#undef ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
using Arcane::MessagePassing::mpAllGather;
using Arcane::MessagePassing::mpAllGatherVariable;
using Arcane::MessagePassing::mpAllReduce;
using Arcane::MessagePassing::mpAllToAll;
using Arcane::MessagePassing::mpAllToAllVariable;
using Arcane::MessagePassing::mpBarrier;
using Arcane::MessagePassing::mpBroadcast;
using Arcane::MessagePassing::mpCreateReceiveSerializeMessage;
using Arcane::MessagePassing::mpCreateRequestListRef;
using Arcane::MessagePassing::mpCreateSendSerializeMessage;
using Arcane::MessagePassing::mpCreateSerializeMessage;
using Arcane::MessagePassing::mpCreateSerializeMessageListRef;
using Arcane::MessagePassing::mpGather;
using Arcane::MessagePassing::mpGatherVariable;
using Arcane::MessagePassing::mpLegacyProbe;
using Arcane::MessagePassing::mpNonBlockingAllGather;
using Arcane::MessagePassing::mpNonBlockingAllReduce;
using Arcane::MessagePassing::mpNonBlockingAllToAll;
using Arcane::MessagePassing::mpNonBlockingAllToAllVariable;
using Arcane::MessagePassing::mpNonBlockingBarrier;
using Arcane::MessagePassing::mpNonBlockingBroadcast;
using Arcane::MessagePassing::mpNonBlockingGather;
using Arcane::MessagePassing::mpProbe;
using Arcane::MessagePassing::mpReceive;
using Arcane::MessagePassing::mpScatterVariable;
using Arcane::MessagePassing::mpSend;
using Arcane::MessagePassing::mpSplit;
using Arcane::MessagePassing::mpTestSome;
using Arcane::MessagePassing::mpWait;
using Arcane::MessagePassing::mpWaitAll;
using Arcane::MessagePassing::mpWaitSome;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
