// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PatchAMR.h                                                  (C) 2000-2025 */
/*                                                                           */
/* TODO                                        */
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/PatchAMRPosition.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PatchAMRPosition::
cellsInPatch(ICartesianMesh* cmesh, UniqueArray<Int32>& cells_local_id) const
{
  VariableNodeReal3& nodes_coord = cmesh->mesh()->nodesCoordinates();
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans la boîte englobante.
  Real3 min_pos = m_position;
  Real3 max_pos = min_pos + m_length;
  //Int32 level = -10;
  cells_local_id.clear();
  ENUMERATE_CELL (icell, cmesh->mesh()->allActiveCells()) {
    Cell cell = *icell;
    Real3 center;
    for (NodeLocalId inode : cell.nodeIds())
      center += nodes_coord[inode];
    center /= cell.nbNode();
    bool is_inside_x = center.x > min_pos.x && center.x < max_pos.x;
    bool is_inside_y = center.y > min_pos.y && center.y < max_pos.y;
    bool is_inside_z = (center.z > min_pos.z && center.z < max_pos.z) || !m_is_3d;
    if (is_inside_x && is_inside_y && is_inside_z) {
      //if (level == -10) level = cell.level();
      //else if (level != cell.level()) ARCANE_FATAL("Level pb"); // TODO plus clair.
      cells_local_id.add(icell.itemLocalId());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
