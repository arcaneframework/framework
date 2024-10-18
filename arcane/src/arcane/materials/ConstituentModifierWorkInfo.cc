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

#include "arcane/materials/internal/ConstituentModifierWorkInfo.h"

#include "arcane/materials/internal/MaterialModifierOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentModifierWorkInfo::
ConstituentModifierWorkInfo(const MemoryAllocationOptions& opts, eMemoryRessource mem)
: pure_local_ids(opts.allocator())
, partial_indexes(opts.allocator())
, cells_changed_in_env(opts)
, cells_unchanged_in_env(opts)
, m_saved_matvar_indexes(opts.allocator())
, m_saved_local_ids(opts.allocator())
, m_cells_current_nb_material(opts)
, m_cells_is_partial(mem)
, m_removed_local_ids_filter(mem)
, m_cells_to_transform(mem)
{
  cells_changed_in_env.setDebugName("WorkInfoCellsChangedInEnv");
  cells_unchanged_in_env.setDebugName("WorkInfoCellsUnchangedInEnv");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentModifierWorkInfo::
initialize(Int32 max_local_id, Int32 nb_material, Int32 nb_environment, RunQueue& queue)
{
  m_is_materials_modified.resizeHost(nb_material);
  m_is_environments_modified.resizeHost(nb_environment);
  m_cells_to_transform.resize(max_local_id);
  m_cells_to_transform.fill(false, &queue);
  m_removed_local_ids_filter.resize(max_local_id);
  m_removed_local_ids_filter.fill(false, &queue);

  m_saved_matvar_indexes.resizeHost(max_local_id);
  m_saved_local_ids.resizeHost(max_local_id);

  // Utilise toujours la mémoire du device pour le tableau contenant les données de copie
  if (queue.isAcceleratorPolicy())
    m_variables_copy_data = NumArray<CopyBetweenPartialAndGlobalOneData, MDDim1>(eMemoryRessource::Device);
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
