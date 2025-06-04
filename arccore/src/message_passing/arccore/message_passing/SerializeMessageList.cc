// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeMessageList.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Liste de messages de sérialisation.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/SerializeMessageList.h"

#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/message_passing/Messages.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/FatalErrorException.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializeMessageList::
SerializeMessageList(IMessagePassingMng* mpm)
: m_message_passing_mng(mpm)
, m_request_list(mpCreateRequestListRef(mpm))
, m_message_passing_phase(timeMetricPhaseMessagePassing(mpm->timeMetricCollector()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializeMessageList::
addMessage(ISerializeMessage* message)
{
  BasicSerializeMessage* true_message = dynamic_cast<BasicSerializeMessage*>(message);
  if (!true_message)
    ARCCORE_FATAL("Can not convert 'ISerializeMessage' to 'BasicSerializeMessage'");
  m_messages_to_process.add(true_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializeMessageList::
processPendingMessages()
{
  for( BasicSerializeMessage* sm : m_messages_to_process ){
    PointToPointMessageInfo message_info(buildMessageInfo(sm));
    bool is_any_source = sm->destination().isNull() || sm->destination().isAnySource();
    if (is_any_source && !m_allow_any_rank_receive) {
      // Il faudra faire un probe pour ce message
      m_messages_to_probe.add({sm,message_info});
    }
    else
      _addMessage(sm,message_info);
    sm->setIsProcessed(true);
  }
  m_messages_to_process.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SerializeMessageList::
waitMessages(eWaitType wait_type)
{
  processPendingMessages();
  // NOTE: il faudrait peut-être faire aussi faire des probe() dans l'appel
  // à _waitMessages() car il est possible que tous les messages n'aient pas
  // été posté. Dans ce cas, il faudrait passer en mode non bloquant tant
  // qu'il y a des probe à faire
  while(!m_messages_to_probe.empty())
    _doProbe();

  return _waitMessages(wait_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializeMessageList::
_doProbe()
{
  // Il faut tester avec probe() si des messages sont disponibles
  for( ProbeInfo& p : m_messages_to_probe ){
    //tm->info() << "CHECK PROBE msg=" << p.m_message_info << " is_done?=" << p.m_is_probe_done;
    // Ne devrait pas être 'vrai' mais par sécurité on fait le test.
    if (p.m_is_probe_done)
      continue;
    MessageId message_id = mpProbe(m_message_passing_mng,p.m_message_info);
    if (message_id.isValid()){
      //tm->info() << "FOUND PROBE message_id=" << message_id;
      PointToPointMessageInfo message_info(message_id,NonBlocking);
      _addMessage(p.m_serialize_message,message_info);
      p.m_is_probe_done = true;
    }
  }

  // Supprime les probes qui sont terminés.
  auto k = std::remove_if(m_messages_to_probe.begin(),m_messages_to_probe.end(),
                          [](const ProbeInfo& p) { return p.m_is_probe_done; });
  m_messages_to_probe.resize(k-m_messages_to_probe.begin());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SerializeMessageList::
_waitMessages(eWaitType wait_type)
{
  TimeMetricSentry tphase(m_message_passing_phase);

  if (wait_type==WaitAll){
    m_request_list->wait(WaitAll);
    // Indique que les messages sont bien terminés
    for( ISerializeMessage* sm : m_messages_serialize )
      sm->setFinished(true);
    m_request_list->clear();
    m_messages_serialize.clear();
    return (-1);
  }

  if (wait_type==WaitSome || wait_type==TestSome){
    Integer nb_request = m_request_list->size();
    m_request_list->wait(wait_type);
    m_remaining_serialize_messages.clear();
    Integer nb_done = 0;
    for( Integer i=0; i<nb_request; ++i ){
      BasicSerializeMessage* sm = m_messages_serialize[i];
      if (m_request_list->isRequestDone(i)){
        ++nb_done;
        sm->setFinished(true);
      }
      else{
        m_remaining_serialize_messages.add(sm);
      }
    }
    m_request_list->removeDoneRequests();
    m_messages_serialize = m_remaining_serialize_messages;
    if (nb_done==nb_request)
      return (-1);
    return nb_done;
  }

  ARCCORE_THROW(NotImplementedException,"waitMessage with wait_type=={0}",(int)wait_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializeMessageList::
_addMessage(BasicSerializeMessage* sm,const PointToPointMessageInfo& message_info)
{
  Request r;
  ISerializer* s = sm->serializer();
  if (sm->isSend())
    r = mpSend(m_message_passing_mng,s,message_info);
  else
    r = mpReceive(m_message_passing_mng,s,message_info);
  m_request_list->add(r);
  m_messages_serialize.add(sm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo SerializeMessageList::
buildMessageInfo(ISerializeMessage* sm)
{
  MessageId message_id(sm->_internalMessageId());
  if (message_id.isValid()){
    PointToPointMessageInfo message_info(message_id,NonBlocking);
    message_info.setEmiterRank(sm->source());
    return message_info;
  }
  return { sm->source(), sm->destination(), sm->internalTag(), NonBlocking }; 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> SerializeMessageList::
createAndAddMessage(MessageRank destination,ePointToPointMessageType type)
{
  auto x = mpCreateSerializeMessage(m_message_passing_mng, destination, type);
  addMessage(x.get());
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
