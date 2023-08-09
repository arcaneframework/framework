// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentModifierWorkInfo.h                                 (C) 2000-2023 */
/*                                                                           */
/* Structure de travail utilisée lors de la modification des constituants.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ComponentModifierWorkInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentModifierWorkInfo::
initialize(Int32 max_local_id)
{
  m_cells_to_transform.resize(max_local_id);
  m_cells_to_transform.fill(false);
  m_removed_local_ids_filter.resize(max_local_id);
  m_removed_local_ids_filter.fill(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentModifierWorkInfo::
setRemovedCells(ConstArrayView<Int32> local_ids,bool value)
{
  // Positionne le filtre des mailles supprimées.
  for (Int32 lid : local_ids)
    m_removed_local_ids_filter[lid] = value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
