// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentModifierWorkInfo.h                               (C) 2000-2024 */
/*                                                                           */
/* Structure de travail utilisée lors de la modification des constituants.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"

#include "arcane/materials/internal/ConstituentModifierWorkInfo.h"

#include "arcane/materials/internal/MaterialModifierOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentModifierWorkInfo::
ConstituentModifierWorkInfo()
: pure_local_ids(platform::getDefaultDataAllocator())
, partial_indexes(platform::getDefaultDataAllocator())
, m_saved_matvar_indexes(platform::getDefaultDataAllocator())
, m_saved_local_ids(platform::getDefaultDataAllocator())
, m_removed_local_ids_filter(platform::getDefaultDataAllocator())
, m_cells_to_transform(platform::getDefaultDataAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentModifierWorkInfo::
initialize(Int32 max_local_id)
{
  m_cells_to_transform.resize(max_local_id);
  m_cells_to_transform.fill(false);
  m_removed_local_ids_filter.resize(max_local_id);
  m_removed_local_ids_filter.fill(false);

  m_saved_matvar_indexes.resize(max_local_id);
  m_saved_local_ids.resize(max_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentModifierWorkInfo::
setRemovedCells(ConstArrayView<Int32> local_ids, bool value)
{
  // Positionne le filtre des mailles supprimées.
  for (Int32 lid : local_ids)
    m_removed_local_ids_filter[lid] = value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentModifierWorkInfo::
setCurrentOperation(MaterialModifierOperation* operation)
{
  m_is_add = operation->isAdd();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
