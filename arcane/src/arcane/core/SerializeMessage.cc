// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeMessage.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Message utilisant un SerializeBuffer.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/SerializeMessage.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arcane::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializeMessage::
SerializeMessage(Int32 orig_rank,Int32 dest_rank,eMessageType mtype)
: BasicSerializeMessage(MessageRank(orig_rank),MessageRank(dest_rank),
                        _toP2PType(mtype),new SerializeBuffer())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializeMessage::
SerializeMessage(Int32 orig_rank,MessageId message_id)
: BasicSerializeMessage(MessageRank(orig_rank),message_id,new SerializeBuffer())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
