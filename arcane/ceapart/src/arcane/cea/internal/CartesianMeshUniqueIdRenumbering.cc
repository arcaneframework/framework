// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.cc                         (C) 2000-2021 */
/*                                                                           */
/* Renumérotation des uniqueId() pour les maillages cartésiens.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cea/internal/CartesianMeshUniqueIdRenumbering.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/ICartesianMeshPatch.h"

#include "arcane/VariableTypes.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include "arcane/core/internal/ICartesianMeshGenerationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshUniqueIdRenumbering::
CartesianMeshUniqueIdRenumbering(ICartesianMesh* cmesh, ICartesianMeshGenerationInfo* gen_info)
: TraceAccessor(cmesh->traceMng())
, m_cartesian_mesh(cmesh)
, m_generation_info(gen_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
renumber()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  Int64 cartesian_global_nb_cell = m_generation_info->globalNbCell();
  info() << "Apply UniqueId renumbering to mesh '" << mesh->name() << "'"
         << " global_nb_cell=" << cartesian_global_nb_cell
         << " global_nb_cell_by_dim=" << m_generation_info->globalNbCells();

  if (mesh->dimension() != 2)
    ARCANE_THROW(NotImplementedException, "Renumbering is only implemented for 2D mesh");

  VariableCellInt32 cells_level(VariableBuildInfo(mesh, "ArcaneRenumberCellsLevel"));
  VariableNodeInt32 nodes_level(VariableBuildInfo(mesh, "ArcaneRenumberNodesLevel"));
  VariableFaceInt32 faces_level(VariableBuildInfo(mesh, "ArcaneRenumberFacesLevel"));

  cells_level.fill(-1);
  nodes_level.fill(-1);
  faces_level.fill(-1);

  // Marque les entités issues du maillage cartésien comme étant de niveau 0
  // Elles ne seront pas renumérotées
  ICartesianMeshPatch* patch0 = m_cartesian_mesh->patch(0);
  ENUMERATE_ (Cell, icell, patch0->cells()) {
    Cell c{ *icell };
    cells_level[icell] = 0;
    for (Node n : c.nodes())
      nodes_level[n] = 0;
    for (Face f : c.faces())
      faces_level[f] = 0;
  }

  // Pour chaque maille de niveau 0, calcule son indice (i,j) dans le maillage cartésien

  // Pour cela, on suppose que le maillage a été créé avec le 'CartesianMeshGenerator'
  // (ou un générateur qui a la même numérotation) et que le uniqueId() d'une maille est:
  //   Int64 cell_unique_id = i + j * all_nb_cell_x;
  // avec:
  //   all_nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];

  Int64 nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];
  if (nb_cell_x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", nb_cell_x);

  ENUMERATE_ (Cell, icell, patch0->cells()) {
    Cell cell{ *icell };
    Int64 uid = cell.uniqueId();
    Int64 coord_i = uid % nb_cell_x;
    Int64 coord_j = uid / nb_cell_x;
    info() << "PARENT_ME: uid=" << cell.uniqueId() << " I=" << coord_i << " J=" << coord_j << " nb_cell_x=" << nb_cell_x;
    _applyChildrenCell(cell, coord_i, coord_j, nb_cell_x, 1, cartesian_global_nb_cell);
  }

  mesh->cellFamily()->notifyItemsUniqueIdChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
_applyChildrenCell(Cell cell, Int64 coord_i, Int64 coord_j, Int64 nb_cell_x, Int32 level, Int64 multiplier)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 4,
  // il faudrait prendre le max des nbHChildren()

  // Suppose qu'on a un pattern 2x2
  coord_i *= 2;
  coord_j *= 2;
  nb_cell_x *= 2;
  multiplier *= 4;

  // Suppose qu'on a 4 enfants comme suit par mailles
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  Int32 nb_child = cell.nbHChildren();
  for (Int32 i = 0; i < nb_child; ++i) {
    Cell sub_cell = cell.hChild(i);
    Int64 my_coord_i = coord_i + i % 2;
    Int64 my_coord_j = coord_j + i / 2;
    Int64 new_uid = (my_coord_i + my_coord_j * nb_cell_x) + multiplier;
    info() << "APPLY_ME: uid=" << sub_cell.uniqueId() << " I=" << my_coord_i << " J=" << my_coord_j
           << " level=" << level << " new_uid=" << new_uid;
    _applyChildrenCell(sub_cell, my_coord_i, my_coord_j, nb_cell_x, level + 1, multiplier);
    sub_cell.internal()->setUniqueId(new_uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

