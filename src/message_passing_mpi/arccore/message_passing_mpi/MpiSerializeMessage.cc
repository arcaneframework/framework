// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeMessage.cc                                      (C) 2000-2020 */
/*                                                                           */
/* Encapsulation de ISerializeMessage pour MPI.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiSerializeMessage.h"

#include "arccore/message_passing/ISerializeMessage.h"
#include "arccore/serialize/BasicSerializer.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeMessage::
MpiSerializeMessage(ISerializeMessage* message,Integer index)
: m_message(message)
, m_serialize_buffer(0)
, m_message_index(index)
, m_message_number(0)
{
  ISerializer* sr = message->serializer();
  m_serialize_buffer = dynamic_cast<BasicSerializer*>(sr);
  if (!m_serialize_buffer){
    ARCCORE_FATAL("Can not convert 'ISerializer' (v={0}) to 'BasicSerializer'",sr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeMessage::
~MpiSerializeMessage()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
