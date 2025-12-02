// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRZonePosition.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Definition d'une zone 2D ou 3D d'un maillage.                             */
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRZonePosition.h"

#include "arcane/utils/FixedArray.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/cartesianmesh/AMRPatchPosition.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRZonePosition::
cellsInPatch(IMesh* mesh, UniqueArray<Int32>& cells_local_id) const
{
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans la boîte englobante.
  Real3 min_pos = m_position;
  Real3 max_pos = min_pos + m_length;
  //Int32 level = -10;
  cells_local_id.clear();
  ENUMERATE_ (Cell, icell, mesh->allActiveCells()) {
    Cell cell = *icell;
    Real3 center;
    for (const Node node : cell.nodes())
      center += nodes_coord[node];
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

void AMRZonePosition::
cellsInPatch(ICartesianMesh* mesh, UniqueArray<Int32>& cells_local_id, AMRPatchPosition& position) const
{
  if (mesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    cellsInPatch(mesh->mesh(), cells_local_id);
    return;
  }
  auto numbering = mesh->_internalApi()->cartesianMeshNumberingMngInternal();

  FixedArray<Int64, 6> min_n_max;
  min_n_max[0] = INT64_MAX;
  min_n_max[1] = INT64_MAX;
  min_n_max[2] = INT64_MAX;
  min_n_max[3] = -1;
  min_n_max[4] = -1;
  min_n_max[5] = -1;
  ArrayView min(min_n_max.view().subView(0, 3));
  ArrayView max(min_n_max.view().subView(3, 3));
  Int64 nb_cells = 0;

  VariableNodeReal3& nodes_coord = mesh->mesh()->nodesCoordinates();
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans la boîte englobante.
  Real3 min_pos = m_position;
  Real3 max_pos = min_pos + m_length;
  Int32 level = -1;
  cells_local_id.clear();
  ENUMERATE_ (Cell, icell, mesh->mesh()->allActiveCells()) {
    Cell cell = *icell;
    Real3 center;
    for (const Node node : cell.nodes())
      center += nodes_coord[node];
    center /= cell.nbNode();
    bool is_inside_x = center.x > min_pos.x && center.x < max_pos.x;
    bool is_inside_y = center.y > min_pos.y && center.y < max_pos.y;
    bool is_inside_z = (center.z > min_pos.z && center.z < max_pos.z) || !m_is_3d;
    if (is_inside_x && is_inside_y && is_inside_z) {
      if (level == -1)
        level = cell.level();
      else if (level != cell.level())
        ARCANE_FATAL("Level pb -- Level recorded before : {0} -- Cell Level : {1} -- CellUID : {2}", level, cell.level(), cell.uniqueId());
      cells_local_id.add(icell.itemLocalId());

      if (icell->isOwn())
        nb_cells++;

      Int64 pos_x = numbering->cellUniqueIdToCoordX(cell);
      if (pos_x < min[MD_DirX])
        min[MD_DirX] = pos_x;
      if (pos_x > max[MD_DirX])
        max[MD_DirX] = pos_x;

      Int64 pos_y = numbering->cellUniqueIdToCoordY(cell);
      if (pos_y < min[MD_DirY])
        min[MD_DirY] = pos_y;
      if (pos_y > max[MD_DirY])
        max[MD_DirY] = pos_y;

      Int64 pos_z = numbering->cellUniqueIdToCoordZ(cell);
      if (pos_z < min[MD_DirZ])
        min[MD_DirZ] = pos_z;
      if (pos_z > max[MD_DirZ])
        max[MD_DirZ] = pos_z;
    }
  }
  mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMin, min);
  mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, max);
  nb_cells = mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, nb_cells);
  Integer level_r = mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, level);

  if (level != -1 && level != level_r) {
    ARCANE_FATAL("Bad level reduced");
  }

  max[MD_DirX] += 1;
  max[MD_DirY] += 1;
  max[MD_DirZ] += 1;

  {
    Int64 nb_cells_patch = (max[MD_DirX] - min[MD_DirX]) * (max[MD_DirY] - min[MD_DirY]) * (max[MD_DirZ] - min[MD_DirZ]);
    if (nb_cells != nb_cells_patch) {
      ARCANE_FATAL("Not regular patch");
    }
  }
  position.setMinPoint({ min[MD_DirX], min[MD_DirY], min[MD_DirZ] });
  position.setMaxPoint({ max[MD_DirX], max[MD_DirY], max[MD_DirZ] });
  position.setLevel(level_r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
