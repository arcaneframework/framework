// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializeMessage.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Message utilisant un BasicSerializeMessage.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ePointToPointMessageType BasicSerializeMessage::
_toP2PType(eMessageType mtype)
{
  switch(mtype){
  case MT_Send: return MsgSend;
  case MT_Recv: return MsgReceive;
  case MT_Broadcast: return MsgSend;
  }
  ARCCORE_FATAL("Unsupported value '{0}'",mtype);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage::eMessageType BasicSerializeMessage::
_toMessageType(ePointToPointMessageType type)
{
  switch(type){
  case MsgSend: return MT_Send;
  case MsgReceive: return MT_Recv;
  }
  ARCCORE_FATAL("Unsupported value '{0}'",type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(MessageRank orig_rank,MessageRank dest_rank,
                      MessageTag tag,ePointToPointMessageType type,
                      BasicSerializer* s)
: m_orig_rank(orig_rank)
, m_dest_rank(dest_rank)
, m_tag(tag)
, m_old_message_type(_toMessageType(type))
, m_message_type(type)
, m_is_send(false)
, m_buffer(s)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(MessageRank orig_rank,MessageRank dest_rank,
                      ePointToPointMessageType type,
                      BasicSerializer* s)
: BasicSerializeMessage(orig_rank,dest_rank,defaultTag(),type,s)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(MessageRank orig_rank,MessageId message_id,
                      BasicSerializer* s)
: m_orig_rank(orig_rank)
, m_dest_rank(message_id.sourceInfo().rank())
, m_tag(message_id.sourceInfo().tag())
, m_old_message_type(MT_Recv)
, m_message_type(MsgReceive)
, m_is_send(false)
, m_buffer(s)
, m_message_id(message_id)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(MessageRank orig_rank,MessageRank dest_rank,
                      MessageTag tag,ePointToPointMessageType type)
: BasicSerializeMessage(orig_rank,dest_rank,tag,type,new BasicSerializer())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(MessageRank orig_rank,MessageRank dest_rank,
                      ePointToPointMessageType type)
: BasicSerializeMessage(orig_rank,dest_rank,defaultTag(),type)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
~BasicSerializeMessage()
{
  delete m_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializeMessage::
_init()
{
  switch(m_old_message_type){
  case MT_Send:
  case MT_Broadcast:
    m_is_send = true;
    break;
  case MT_Recv:
    m_is_send = false;
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicSerializeMessage::
setStrategy(eStrategy strategy)
{
  if (m_is_processed)
    ARCCORE_FATAL("Can not change strategy if isProcessed() is true");
  m_strategy = strategy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> BasicSerializeMessage::
create(MessageRank source,MessageRank destination,MessageTag tag,
       ePointToPointMessageType type)
{
  ISerializeMessage* m = new BasicSerializeMessage(source,destination,tag,type);
  return makeRef(m);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> BasicSerializeMessage::
create(MessageRank source,MessageRank destination,ePointToPointMessageType type)
{
  ISerializeMessage* m = new BasicSerializeMessage(source,destination,type);
  return makeRef(m);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> BasicSerializeMessage::
create(MessageRank source,MessageId message_id)
{
  ISerializeMessage* m = new BasicSerializeMessage(source,message_id,new BasicSerializer());
  return makeRef(m);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
