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
/* BasicSerializeMessage.cc                                    (C) 2000-2020 */
/*                                                                           */
/* Message utilisant un BasicSerializeMessage.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
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
                      ePointToPointMessageType type,
                      BasicSerializer* s)
: m_orig_rank(orig_rank)
, m_dest_rank(dest_rank)
, m_tag(defaultTag())
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
                      ePointToPointMessageType type)
: BasicSerializeMessage(orig_rank,dest_rank,type,new BasicSerializer())
{
  _init();
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

Ref<ISerializeMessage> BasicSerializeMessage::
create(MessageRank source,MessageRank destination,ePointToPointMessageType type)
{
  ISerializeMessage* m = new BasicSerializeMessage(source,destination,type);
  return makeRef(m);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
