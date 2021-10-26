// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleGridMeshPartitioner.cc                                (C) 2000-2021 */
/*                                                                           */
/* Partitionneur de maillage sur une grille.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/IGridMeshPartitioner.h"
#include "arcane/BasicService.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/ServiceFactory.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"

#include "arcane/IMeshPartitionConstraintMng.h"
#include "arcane/IMeshUtilities.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partitionneur de maillage sur une grille.
 */
class SimpleGridMeshPartitioner
: public BasicService
, public IGridMeshPartitioner
{
 public:
  explicit SimpleGridMeshPartitioner(const ServiceBuildInfo& sbi);

 public:
  void build() override {}
  IPrimaryMesh* primaryMesh() { return mesh()->toPrimaryMesh(); }
  void partitionMesh(bool initial_partition) override;
  void notifyEndPartition() override {}

 public:
  void setBoundingBox(Real3 min_val, Real3 max_val) override
  {
    m_min_box = min_val;
    m_max_box = max_val;
    m_is_bounding_box_set = true;
  }
  void setPartIndex(Int32 i, Int32 j, Int32 k) override
  {
    m_ijk_part[0] = i;
    m_ijk_part[1] = j;
    m_ijk_part[2] = k;
    m_is_ijk_set = true;
  }

 private:
  Real3 m_min_box;
  Real3 m_max_box;
  std::array<Int32, 3> m_ijk_part;
  bool m_is_bounding_box_set = false;
  bool m_is_ijk_set = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleGridMeshPartitioner::
SimpleGridMeshPartitioner(const ServiceBuildInfo& sbi)
: BasicService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
partitionMesh([[maybe_unused]] bool initial_partition)
{
  if (!m_is_bounding_box_set)
    ARCANE_FATAL("Bounding box is not set. call method setBoundingBox() before");
  if (!m_is_ijk_set)
    ARCANE_FATAL("Part index is not set. call method setPartIndex() before");
  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();

  IParallelMng* pm = mesh->parallelMng();
  Int32 nb_rank = pm->commSize();

  // Calcule le nombre de parties par direction
  std::array<Int32, 3> nb_part_by_direction_buf = { 0, 0, 0 };
  ArrayView<Int32> nb_part_by_direction(3, nb_part_by_direction_buf.data());
  for (Integer i = 0; i < 3; ++i)
    if (m_ijk_part[i] >= 0)
      nb_part_by_direction[i] = m_ijk_part[i] + 1;

  Int32 nb_direction = 0;
  if (nb_part_by_direction[2] > 0)
    nb_direction = 3;
  else if (nb_part_by_direction[1] > 0)
    nb_direction = 2;
  else if (nb_part_by_direction[0] > 0)
    ARCANE_FATAL("No part");

  pm->reduce(Parallel::ReduceMax, nb_part_by_direction);
  info() << "NB_DIRECTION=" << nb_direction << " NB_PART=" << nb_part_by_direction;

  // Calcul les coordonnées de la grille par direction
  const Real min_value = -FloatInfo<Real>::maxValue();
  UniqueArray<UniqueArray<Real>> grid_coord(nb_direction);
  for (Integer i = 0; i < nb_direction; ++i)
    grid_coord[i].resize(nb_part_by_direction[i], min_value);

  for (Integer i = 0; i < nb_direction; ++i) {
    Int32 index = m_ijk_part[i];
    if (index >= 0)
      grid_coord[i][index] = m_min_box[i];
  }
  for (Integer i = 0; i < nb_direction; ++i) {
    pm->reduce(Parallel::ReduceMax, grid_coord[i]);
    info() << "GRID_COORD dir=" << i << " V=" << grid_coord[i];
  }

  // Vérifie que chaque direction de la grille est triée.
  for (Integer i = 0; i < nb_direction; ++i) {
    ConstArrayView<Real> coords(grid_coord[0]);
    Int32 nb_value = coords.size();
    if (nb_value == 0)
      continue;
    for (Int32 z = 0; z < (nb_value - 1); ++z)
      if (coords[z] > coords[z + 1])
        ARCANE_FATAL("Grid coord '{0}' is not sorted: {1} > {2}", i, coords[z], coords[z + 1]);
  }

  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);

  const Int32 offset_y = (nb_direction >= 2) ? nb_part_by_direction[0] : 0;
  const Int32 offset_z = (nb_direction == 3) ? (nb_part_by_direction[0] * nb_part_by_direction[1]) : 0;

  // Parcours pour chaque maille chaque direction et regarder dans indice dans
  // la grille elle se trouve. En déduit le nouveau rang.
  ENUMERATE_ (Cell, icell, mesh->allCells().own()) {
    Cell cell = *icell;
    std::array<Int32, 3> cell_part = { -1, -1, -1 };
    Real3 cell_center;
    Int32 nb_node = cell.nbNode();
    for (Integer inode = 0; inode < nb_node; ++inode)
      cell_center += nodes_coord[cell.node(inode)];
    cell_center /= static_cast<Real>(nb_node);
    for (Integer idir = 0; idir < nb_direction; ++idir) {
      ConstArrayView<Real> coords(grid_coord[idir]);
      Int32 nb_value = coords.size();
      // TODO: utiliser une dichotomie
      Real cc = cell_center[idir];
      for (Int32 z = 0; z < (nb_value - 1); ++z) {
        if (cc < coords[z + 1])
          cell_part[idir] = z;
      }
      if (cell_part[idir] == (-1))
        cell_part[idir] = (nb_value - 1);
    }
    Int32 new_owner = cell_part[0] + cell_part[1] * offset_y + cell_part[2] * offset_z;
    info() << "CELL=" << ItemPrinter(cell) << " coord=" << cell_center << " new_owner=" << new_owner
           << " dir=" << cell_part[0] << " " << cell_part[1] << " " << cell_part[2];
    if (new_owner < 0 || new_owner >= nb_rank)
      ARCANE_FATAL("Bad value for new owner cell={0} new_owner={1} (max={2})", ItemPrinter(cell), new_owner, nb_rank);
    cells_new_owner[icell] = new_owner;
  }

  cells_new_owner.synchronize();
  if (mesh->partitionConstraintMng()) {
    // Deal with Tied Cells
    mesh->partitionConstraintMng()->computeAndApplyConstraints();
  }
  mesh->utilities()->changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SimpleGridMeshPartitioner,
                        ServiceProperty("SimpleGridMeshPartitioner", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IGridMeshPartitioner));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
