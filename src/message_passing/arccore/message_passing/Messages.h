// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Messages.h                                                  (C) 2000-2020 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGES_H
#define ARCCORE_MESSAGEPASSING_MESSAGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IDispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(type)                                                               \
  inline void mpAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf)                     \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->allGather(send_buf, recv_buf);                                                  \
  }                                                                                                                   \
  inline void mpGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank)            \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->gather(send_buf, recv_buf, rank);                                               \
  }                                                                                                                   \
  inline void mpAllGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf)           \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->allGatherVariable(send_buf, recv_buf);                                          \
  }                                                                                                                   \
  inline void mpGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf, Int32 rank)  \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->gatherVariable(send_buf, recv_buf, rank);                                       \
  }                                                                                                                   \
  inline void mpScatterVariable(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 root);  \
  inline type mpAllReduce(IMessagePassingMng* pm, eReduceType rt, type v)                                             \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->allReduce(rt, v);                                                        \
  }                                                                                                                   \
  inline void mpAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<type> v)                                       \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->allReduce(rt, v);                                                               \
  }                                                                                                                   \
  inline void mpBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank)                                    \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->broadcast(send_buf, rank);                                                      \
  }                                                                                                                   \
  inline void mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank)                                     \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->send(values, rank, true);                                                       \
  }                                                                                                                   \
  inline void mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank)                                        \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    pm->dispatchers()->dispatcher(x)->receive(values, rank, true);                                                    \
  }                                                                                                                   \
  inline Request mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank, bool is_blocked)                 \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->send(values, rank, is_blocked);                                          \
  }                                                                                                                   \
  inline Request mpSend(IMessagePassingMng* pm, Span<const type> values, PointToPointMessageInfo message)             \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->send(values, message);                                                   \
  }                                                                                                                   \
  inline Request mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank, bool is_blocked)                    \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->receive(values, rank, is_blocked);                                       \
  }                                                                                                                   \
  inline Request mpReceive(IMessagePassingMng* pm, Span<type> values, PointToPointMessageInfo message)                \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->receive(values, message);                                                \
  }                                                                                                                   \
  inline void mpAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count)         \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    return pm->dispatchers()->dispatcher(x)->allToAll(send_buf, recv_buf, count);                                     \
  }                                                                                                                   \
  inline void mpAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                                 ConstArrayView<Int32> send_index, Span<type> recv_buf,                               \
                                 ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index)                  \
  {                                                                                                                   \
    type* x = nullptr;                                                                                                \
    auto d = pm->dispatchers()->dispatcher(x);                                                                        \
    d->allToAllVariable(send_buf, send_count, send_index, recv_buf, recv_count, recv_index);                          \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpWaitAll(IMessagePassingMng* pm, ArrayView<Request> requests)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitAllRequests(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpWait(IMessagePassingMng* pm, Request request)
{
  mpWaitAll(pm, ArrayView<Request>(1, &request));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpWaitSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpTestSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpWait(IMessagePassingMng* pm, ArrayView<Request> requests,
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
/*!
 * \brief Teste si un message est disponible.
 *
 * Cette fonction permet de savoir si un message issu du couple (rang,tag)
 * est disponible. \a message doit avoir été initialisé avec un couple (rang,tag)
 * (message.isRankTag() doit être vrai).
 * Retourne une instance de \a MessageId.
 * En mode non bloquant, si aucun message n'est disponible, alors
 * MessageId::isValid() vaut \a false pour l'instance retournée.
 */
inline MessageId
mpProbe(IMessagePassingMng* pm, PointToPointMessageInfo message)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->probe(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline IMessagePassingMng*
mpSplit(IMessagePassingMng* pm, bool keep)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->commSplit(keep);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
mpBarrier(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->barrier();
}

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
