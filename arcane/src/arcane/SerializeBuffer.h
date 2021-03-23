// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeBuffer.h                                           (C) 2000-2020 */
/*                                                                           */
/* Tampon de serialisation.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERIALIZEBUFFER_H
#define ARCANE_SERIALIZEBUFFER_H
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
 */
class ARCANE_CORE_EXPORT SerializeBuffer
: public Arccore::BasicSerializer
{
 public:
  void allGather(IParallelMng* pm,const SerializeBuffer& send_serializer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
