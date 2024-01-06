// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeBuffer.h                                           (C) 2000-2024 */
/*                                                                           */
/* Tampon de serialisation.                                                  */
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
 * \brief Implémentation d'un tampon pour la sérialisation.
 *
 * Cette classe est interne à %Arcane et ne doit pas être utilisée en
 * dehors.
 *
 * Cette classe est obsolète. Il faut Utiliser Arccore::BasicSerializer
 * à la place
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
