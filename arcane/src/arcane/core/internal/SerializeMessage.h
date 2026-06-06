// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeMessage.h                                          (C) 2000-2025 */
/*                                                                           */
/* Message using a SerializeBuffer.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_SERIALIZEMESSAGE_H
#define ARCANE_CORE_INTERNAL_SERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Message using a SerializeBuffer.
 *
 * A message consists of a series of bytes sent from one rank
 * (origRank()) to another (destRank()). If isSend() is true,
 * origRank() sends to destRank(), otherwise it is the reverse.
 * If it is a receive message, the serializer() is allocated
 * and filled automatically.
 *
 * For parallelism to work correctly, a complementary message
 * must be sent by destRank().
 */
class ARCANE_CORE_EXPORT SerializeMessage
: public MessagePassing::internal::BasicSerializeMessage
{
 public:

  SerializeMessage(Int32 orig_rank, Int32 dest_rank, eMessageType mtype);
  SerializeMessage(Int32 orig_rank, MessagePassing::MessageId message_id);

 public:

  ARCCORE_DEPRECATED_2020("Use BasicSerializeMessage::serializer() instead")
  SerializeBuffer& buffer()
  {
    // Since this class created the serializer(), we are certain
    // that the conversion is valid.
    Arccore::BasicSerializer& x = MessagePassing::internal::BasicSerializeMessage::buffer();
    return static_cast<SerializeBuffer&>(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
