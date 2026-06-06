// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeBuffer.h                                           (C) 2000-2024 */
/*                                                                           */
/* Serialization buffer.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERIALIZEBUFFER_H
#define ARCANE_CORE_SERIALIZEBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arccore/serialize/BasicSerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Implementation of a buffer for serialization.
 *
 * This class is internal to %Arcane and should not be used
 * externally.
 *
 * This class is obsolete. You must use Arccore::BasicSerializer
 * instead.
 */
class ARCANE_CORE_EXPORT SerializeBuffer
: public Arccore::BasicSerializer
{
 public:

  ARCANE_DEPRECATED_REASON("Y2024: Use mpAllGather() instead")
  void allGather(IParallelMng* pm, const SerializeBuffer& send_serializer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
