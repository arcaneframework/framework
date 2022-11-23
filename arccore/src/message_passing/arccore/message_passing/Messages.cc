// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Messages.cc                                                 (C) 2000-2022 */
/*                                                                           */
/* Identifiant d'un message point à point.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Messages.h"

#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/MessageId.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"

#include "arccore/base/NotSupportedException.h"

/*!
 * \file Messages.h
 *
 * \brief Liste des fonctions d'échange de message.
 */
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
  d->waitSomeRequests(requests, indexes, true);
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

void
mpBarrier(IMessagePassingMng* pm)
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
  return d->sendSerializer(values,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request
mpReceive(IMessagePassingMng* pm, ISerializer* values,
          const PointToPointMessageInfo& message)
{
  auto d = pm->dispatchers()->serializeDispatcher();
  return d->receiveSerializer(values,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo IControlDispatcher::
legacyProbe(const PointToPointMessageInfo&)
{
  ARCCORE_THROW(NotSupportedException,"pure virtual call to legacyProbe()");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
