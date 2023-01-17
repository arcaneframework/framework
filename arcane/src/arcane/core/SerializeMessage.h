﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeMessage.h                                          (C) 2000-2020 */
/*                                                                           */
/* Message utilisant un SerializeBuffer.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERIALIZEMESSAGE_H
#define ARCANE_SERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arcane/SerializeBuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arccore::MessagePassing;
using namespace Arccore::MessagePassing::internal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Message utilisant un SerializeBuffer.
 *
 * Un message consiste en une série d'octets envoyés d'un rang
 * (origRank()) à un autre (destRank()). Si isSend() est vrai,
 * c'est origRank() qui envoie à destRank(), sinon c'est l'inverse.
 * S'il s'agit d'un message de réception, le serializer() est alloué
 * et remplit automatiquement.
 *
 * Pour que le parallélisme fonctionne correctement, il faut qu'un message
 * complémentaire à celui-ci soit envoyé par destRank().
 */
class ARCANE_CORE_EXPORT SerializeMessage
: public BasicSerializeMessage
{
 public:
  SerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype);
  SerializeMessage(Int32 orig_rank,MessageId message_id);
 public:
  ARCCORE_DEPRECATED_2020("Use BasicSerializeMessage::serializer() instead")
  SerializeBuffer& buffer()
  {
    // Comme c'est cette classe qui a créé le serializer(), on est certain
    // que la conversion est valide.
    Arccore::BasicSerializer& x = BasicSerializeMessage::buffer();
    return static_cast<SerializeBuffer&>(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

