// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointSerializerMng.cc                                (C) 2000-2020 */
/*                                                                           */
/* Communications point à point par des 'ISerializer'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/PointToPointSerializerMng.h"

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
using internal::BasicSerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PointToPointSerializerMng::Impl
{
 public:
  Impl(IMessagePassingMng* mpm)
  : m_message_passing_mng(mpm), m_rank(mpm->commRank())
  {
    m_message_list = mpCreateSerializeMessageListRef(mpm);
  }
 public:
  void addMessage(Ref<ISerializeMessage> message)
  {
    message->setStrategy(m_strategy);
    m_pending_messages.add(message);
  }
  void processPendingMessages()
  {
    for( auto& x : m_pending_messages ){
      m_message_list->addMessage(x.get());
      m_waiting_messages.add(x);
    }
    m_pending_messages.clear();
  }
  Integer waitMessages(eWaitType wt,std::function<void(ISerializeMessage*)> functor)
  {
    processPendingMessages();
    Integer n = m_message_list->waitMessages(wt);
    UniqueArray<Ref<ISerializeMessage>> new_waiting_messages;
    for( auto& x : m_waiting_messages ){
      if (x->finished()){
        functor(x.get());
      }
      else
        new_waiting_messages.add(x);
    }
    m_waiting_messages.clear();
    m_waiting_messages.copy(new_waiting_messages);
    return n;
  }
  bool hasMessages() const { return !m_pending_messages.empty() || !m_waiting_messages.empty(); }
 public:
  IMessagePassingMng* m_message_passing_mng;
  MessageRank m_rank;
  Ref<ISerializeMessageList> m_message_list;
  ISerializeMessage::eStrategy m_strategy = ISerializeMessage::eStrategy::Default;
  MessageTag m_tag = BasicSerializeMessage::defaultTag();
 private:
  UniqueArray<Ref<ISerializeMessage>> m_pending_messages;
  UniqueArray<Ref<ISerializeMessage>> m_waiting_messages;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointSerializerMng::
PointToPointSerializerMng(IMessagePassingMng* mpm)
: m_p(new Impl(mpm))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointSerializerMng::
~PointToPointSerializerMng()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng* PointToPointSerializerMng::
messagePassingMng() const
{
  return m_p->m_message_passing_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PointToPointSerializerMng::
processPendingMessages()
{
  m_p->processPendingMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer PointToPointSerializerMng::
waitMessages(eWaitType wt,std::function<void(ISerializeMessage*)> functor)
{
  return m_p->waitMessages(wt,functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PointToPointSerializerMng::
hasMessages() const
{
  return m_p->hasMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PointToPointSerializerMng::
setDefaultTag(MessageTag default_tag)
{
  if (hasMessages())
    ARCCORE_FATAL("Can not call setDefaultTag() if hasMessages()==true");
  m_p->m_tag = default_tag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PointToPointSerializerMng::
setStrategy(ISerializeMessage::eStrategy strategy)
{
  if (hasMessages())
    ARCCORE_FATAL("Can not call setStrategy() if hasMessages()==true");
  m_p->m_strategy = strategy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> PointToPointSerializerMng::
addSendMessage(MessageRank receiver_rank)
{
  auto x = BasicSerializeMessage::create(m_p->m_rank,receiver_rank,m_p->m_tag,MsgSend);
  m_p->addMessage(x);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> PointToPointSerializerMng::
addReceiveMessage(MessageRank sender_rank)
{
  auto x = BasicSerializeMessage::create(m_p->m_rank,sender_rank,m_p->m_tag,MsgReceive);
  m_p->addMessage(x);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> PointToPointSerializerMng::
addReceiveMessage(MessageId message_id)
{
  auto x = BasicSerializeMessage::create(m_p->m_rank,message_id);
  m_p->addMessage(x);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
