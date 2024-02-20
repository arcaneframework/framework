// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshTimeHistoryAdder.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Classe permettant d'ajouter un historique de valeur lié à un maillage.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshTimeHistoryAdder.h"
#include "arcane/core/internal/ITimeHistoryMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshTimeHistoryAdder::
MeshTimeHistoryAdder(ITimeHistoryMng* thm, const MeshHandle& mesh_handle)
: m_thm(thm)
, m_mesh_handle(mesh_handle)
{}

void MeshTimeHistoryAdder::
addValue(const TimeHistoryAddValueArg& thp, Real value)
{
  m_thm->_internalApi()->addValue(TimeHistoryAddValueArgInternal(thp, m_mesh_handle), value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
