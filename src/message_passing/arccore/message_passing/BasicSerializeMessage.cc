// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* BasicSerializeMessage.cc                                    (C) 2000-2020 */
/*                                                                           */
/* Message utilisant un BasicSerializeMessage.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/BasicSerializeMessage.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializeMessage::
BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype)
: m_orig_rank(orig_rank)
, m_dest_rank(dest_rank)
, m_tag(0)
, m_message_type(mtype)
, m_is_send(false)
, m_finished(false)
{
  switch(mtype){
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

BasicSerializeMessage::
BasicSerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype,
                      BasicSerializer* s)
: BasicSerializeMessage(orig_rank,dest_rank,mtype)
{
  m_buffer = s;
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

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
