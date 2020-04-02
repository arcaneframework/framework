// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Messages.cc                                                 (C) 2000-2020 */
/*                                                                           */
/* Identifiant d'un message point à point.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Messages.h"

#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
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

void
mpWaitAll(IMessagePassingMng* pm, ArrayView<Request> requests)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitAllRequests(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
mpWait(IMessagePassingMng* pm, Request request)
{
  mpWaitAll(pm, ArrayView<Request>(1, &request));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
mpWaitSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
mpTestSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->waitSomeRequests(requests, indexes, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
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

MessageId
mpProbe(IMessagePassingMng* pm, PointToPointMessageInfo message)
{
  auto d = pm->dispatchers()->controlDispatcher();
  return d->probe(message);
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

void
mpBarrier(IMessagePassingMng* pm)
{
  auto d = pm->dispatchers()->controlDispatcher();
  d->barrier();
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
