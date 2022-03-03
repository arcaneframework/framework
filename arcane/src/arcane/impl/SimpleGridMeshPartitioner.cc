// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleGridMeshPartitioner.cc                                (C) 2000-2022 */
/*                                                                           */
/* Partitionneur de maillage sur une grille.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/IGridMeshPartitioner.h"
#include "arcane/BasicService.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/ServiceFactory.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IExtraGhostCellsBuilder.h"

#include "arcane/IMeshPartitionConstraintMng.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IItemFamily.h"

#include <array>
#include <map>

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

  /*!
   * \brief Informations sur les mailles fantômes supplémentaires.
   *
   * Il faut conserver les uniqueId() lors de la contruction et les transformer
   * en localId() uniquement dans computeExtraCellsToSend() car durant
   * le partitionnement les localId() peuvent changer.
   *
   * \note Pour l'instant on ne peut pas détruire les instances de cette classe
   * car on ne peut pas supprimer les références enregistrées dans IMeshModifier.
   */
  class GhostCellsBuilder
  : public IExtraGhostCellsBuilder
  {
   public:

    explicit GhostCellsBuilder(IMesh* mesh)
    : m_mesh(mesh)
    {}

   public:

    void computeExtraCellsToSend() override
    {
      for (auto v : m_ghost_cell_uids) {
        Int32 rank = v.first;
        Int32 nb_ghost = v.second.size();
        UniqueArray<Int32>& local_ids = m_ghost_cell_local_ids[rank];
        local_ids.resize(nb_ghost);
        m_mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids, v.second);
      }
    }

    Int32ConstArrayView extraCellsToSend(Int32 rank) const override
    {
      auto x = m_ghost_cell_local_ids.find(rank);
      if (x == m_ghost_cell_local_ids.end())
        return {};
      return x->second;
    }

    std::map<Int32, UniqueArray<ItemUniqueId>> m_ghost_cell_uids;
    std::map<Int32, UniqueArray<Int32>> m_ghost_cell_local_ids;
    IMesh* m_mesh;
  };

  class GridInfo
  {
   public:

    UniqueArray<UniqueArray<Real>> m_grid_coord;
    Int32 m_nb_direction = 0;
    Int32 m_offset_y = 0;
    Int32 m_offset_z = 0;
  };

 public:

  explicit SimpleGridMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  void build() override {}
  IPrimaryMesh* primaryMesh() override { return mesh()->toPrimaryMesh(); }
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

  void applyMeshPartitioning(IMesh* mesh) override;

 private:

  Real3 m_min_box;
  Real3 m_max_box;
  std::array<Int32, 3> m_ijk_part;
  bool m_is_bounding_box_set = false;
  bool m_is_ijk_set = false;
  bool m_is_verbose = false;
  GhostCellsBuilder* m_ghost_cells_builder = nullptr;
  ScopedPtrT<GridInfo> m_grid_info;

 private:

  Int32 _findPart(RealConstArrayView coords, Real center);
  void _addGhostCell(Int32 rank, Cell cell);
  void _buildGridInfo();
  void _computeSpecificGhostLayer();
  void _addCellToIntersectedParts(Cell cell, std::array<Int32, 3> min_part, std::array<Int32, 3> nb_part);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleGridMeshPartitioner::
SimpleGridMeshPartitioner(const ServiceBuildInfo& sbi)
: BasicService(sbi)
{
  if (platform::getEnvironmentVariable("ARCANE_DEBUG_SIMPLE_GRID_MESH_PARTITIONER") == "1")
    m_is_verbose = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'indice dans \a coords de la valeur immédiatement inférieure
 * à \a position.
 *
 * Le tableau \a coords doit être trié par ordre croissant.
 * // TODO: utiliser une dichotomie.
 */
Int32 SimpleGridMeshPartitioner::
_findPart(RealConstArrayView coords, Real position)
{
  const Int32 nb_value = coords.size();
  if (position < coords[0])
    return 0;

  Int32 part_id = -1;
  for (Int32 z = 0; z < nb_value; ++z) {
    if (m_is_verbose)
      info() << " z=" << z << " coord=" << coords[z] << " part=" << part_id;
    if (position > coords[z])
      part_id = z;
    else
      break;
  }

  if (part_id == (-1))
    part_id = (nb_value - 1);

  return part_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
_addGhostCell(Int32 rank, Cell cell)
{
  m_ghost_cells_builder->m_ghost_cell_uids[rank].add(cell.uniqueId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
_buildGridInfo()
{
  m_grid_info = new GridInfo();

  if (!m_is_bounding_box_set)
    ARCANE_FATAL("Bounding box is not set. call method setBoundingBox() before");
  if (!m_is_ijk_set)
    ARCANE_FATAL("Part index is not set. call method setPartIndex() before");

  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
  const Int32 dimension = mesh->dimension();

  IParallelMng* pm = mesh->parallelMng();
  //Int32 nb_rank = pm->commSize();

  // Calcule le nombre de parties par direction
  std::array<Int32, 3> nb_part_by_direction_buf = { 0, 0, 0 };
  ArrayView<Int32> nb_part_by_direction(3, nb_part_by_direction_buf.data());
  for (Integer i = 0; i < 3; ++i)
    if (m_ijk_part[i] >= 0)
      nb_part_by_direction[i] = m_ijk_part[i] + 1;

  auto& nb_direction = m_grid_info->m_nb_direction;

  nb_direction = 0;
  if (nb_part_by_direction[2] > 0)
    nb_direction = 3;
  else if (nb_part_by_direction[1] > 0)
    nb_direction = 2;
  else
    ARCANE_THROW(NotImplementedException, "SimpleGridMeshPartitioner for 1D mesh");

  if (nb_direction != dimension)
    ARCANE_FATAL("Invalid number of direction: mesh_dimension={0} nb_direction={1}", dimension, nb_direction);

  pm->reduce(Parallel::ReduceMax, nb_part_by_direction);
  info() << "NB_DIRECTION=" << nb_direction << " NB_PART=" << nb_part_by_direction;

  // Calcul les coordonnées de la grille par direction
  const Real min_value = -FloatInfo<Real>::maxValue();
  auto& grid_coord = m_grid_info->m_grid_coord;

  grid_coord.resize(nb_direction);
  grid_coord.resize(nb_direction);
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

  // Vérifie que chaque direction de la grille est croissante
  for (Integer i = 0; i < nb_direction; ++i) {
    ConstArrayView<Real> coords(grid_coord[0].view());
    Int32 nb_value = coords.size();
    if (nb_value == 0)
      continue;
    for (Int32 z = 0; z < (nb_value - 1); ++z)
      if (coords[z] > coords[z + 1])
        ARCANE_FATAL("Grid coord '{0}' is not sorted: {1} > {2}", i, coords[z], coords[z + 1]);
  }

  m_grid_info->m_offset_y = (nb_direction >= 2) ? nb_part_by_direction[0] : 0;
  m_grid_info->m_offset_z = (nb_direction == 3) ? (nb_part_by_direction[0] * nb_part_by_direction[1]) : 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
partitionMesh([[maybe_unused]] bool initial_partition)
{
  if (m_grid_info.get())
    ARCANE_FATAL("partitionMesh() has already been called. Only one call par SimpleGridMeshPartitioner instance is allowed");

  _buildGridInfo();

  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
  IParallelMng* pm = mesh->parallelMng();
  Int32 nb_rank = pm->commSize();

  const Int32 offset_y = m_grid_info->m_offset_y;
  const Int32 offset_z = m_grid_info->m_offset_z;
  const Int32 nb_direction = m_grid_info->m_nb_direction;

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
      ConstArrayView<Real> coords(m_grid_info->m_grid_coord[idir].view());
      Real cc = cell_center[idir];

      if (m_is_verbose)
        info() << " Cell uid=" << cell.uniqueId() << " idir=" << idir << " cc=" << cc;

      cell_part[idir] = _findPart(coords, cc);
    }

    const Int32 new_owner = cell_part[0] + cell_part[1] * offset_y + cell_part[2] * offset_z;
    if (m_is_verbose)
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
/*!
 * Parcours toutes les parties qui intersectent la maille \a cell et envoie
 * cette maille en tant que fantôme.
 */
void SimpleGridMeshPartitioner::
_addCellToIntersectedParts(Cell cell, std::array<Int32, 3> min_part, std::array<Int32, 3> nb_part)
{
  const Int32 offset_y = m_grid_info->m_offset_y;
  const Int32 offset_z = m_grid_info->m_offset_z;
  const Int32 cell_owner = cell.owner();

  for (Integer k0 = 0, maxk0 = nb_part[0]; k0 < maxk0; ++k0)
    for (Integer k1 = 0, maxk1 = nb_part[1]; k1 < maxk1; ++k1)
      for (Integer k2 = 0, maxk2 = nb_part[2]; k2 < maxk2; ++k2) {
        Int32 p0 = min_part[0] + k0;
        Int32 p1 = min_part[1] + k1;
        Int32 p2 = min_part[2] + k2;
        Int32 owner = p0 + (p1 * offset_y) + (p2 * offset_z);
        // On ne s'envoie pas à soi-même.
        if (owner != cell_owner)
          _addGhostCell(owner, cell);
      }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
_computeSpecificGhostLayer()
{
  if (!m_grid_info.get())
    ARCANE_FATAL("partitionMesh() has to be called before this method.");

  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();

  const Int32 nb_direction = m_grid_info->m_nb_direction;

  ENUMERATE_ (Cell, icell, mesh->allCells().own()) {
    Cell cell = *icell;

    std::array<Int32, 3> min_part = { -1, -1, -1 };
    std::array<Int32, 3> max_part = { -1, -1, -1 };
    std::array<Int32, 3> nb_node_part = { 1, 1, 1 };

    for (Node node : cell.nodes()) {
      std::array<Int32, 3> node_part = { -1, -1, -1 };
      Real3 node_position = nodes_coord[node];

      for (Integer idir = 0; idir < nb_direction; ++idir) {
        ConstArrayView<Real> coords(m_grid_info->m_grid_coord[idir].view());
        Real cc = node_position[idir];

        if (m_is_verbose)
          info() << " Node uid=" << node.uniqueId() << " idir=" << idir << " cc=" << cc;
        Int32 part_id = _findPart(coords, cc);
        if (m_is_verbose)
          info() << " Node uid=" << node.uniqueId() << " idir=" << idir << " part=" << node_part[idir];

        // Initialise le min/max si pas encore fait
        if (min_part[idir] == (-1))
          min_part[idir] = part_id;
        if (max_part[idir] == (-1))
          max_part[idir] = part_id;

        // Met à jour le min/max si pas encore fait
        if (min_part[idir] > part_id)
          min_part[idir] = part_id;
        if (max_part[idir] < part_id)
          max_part[idir] = part_id;

        node_part[idir] = part_id;
      }

      if (m_is_verbose)
        info() << " ** Node part uid=" << node.uniqueId() << " part=" << ArrayView<Int32>(node_part);
    }

    Int32 total_nb_part = 1;
    for (Integer idir = 0; idir < nb_direction; ++idir) {
      Int32 nb_part = 1 + (max_part[idir] - min_part[idir]);
      nb_node_part[idir] = nb_part;
      total_nb_part *= nb_part;
    }

    if (m_is_verbose)
      info() << " Cell uid=" << cell.uniqueId() << " min_part=" << ArrayView<Int32>(min_part)
             << " max_part=" << ArrayView<Int32>(max_part)
             << " nb_part=" << ArrayView<Int32>(nb_node_part)
             << " total=" << total_nb_part;

    _addCellToIntersectedParts(cell, min_part, nb_node_part);
  }

  if (m_is_verbose) {
    info() << "GHOST_CELLS_TO_SEND";
    for (auto v : m_ghost_cells_builder->m_ghost_cell_uids) {
      info() << "RANK=" << v.first << " ids=" << v.second;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleGridMeshPartitioner::
applyMeshPartitioning(IMesh* mesh)
{
  //TODO A terme supprimer l'utilisation du maillage issu de l'instance.
  if (mesh != this->mesh())
    ARCANE_FATAL("mesh argument should be the same mesh that the one used to create this instance");

  if (m_ghost_cells_builder)
    ARCANE_FATAL("Only one call to this method is allower per instance.");

  mesh->modifier()->setDynamic(true);
  mesh->utilities()->partitionAndExchangeMeshWithReplication(this, true);

  ScopedPtrT<GhostCellsBuilder> scoped_builder{ new GhostCellsBuilder(mesh) };
  m_ghost_cells_builder = scoped_builder.get();
  mesh->modifier()->addExtraGhostCellsBuilder(m_ghost_cells_builder);

  // Recalcule spécifiquement les mailles fantômes pour recouvrir les partitions
  _computeSpecificGhostLayer();
  mesh->modifier()->updateGhostLayers();

  mesh->modifier()->removeExtraGhostCellsBuilder(m_ghost_cells_builder);

  m_ghost_cells_builder = nullptr;
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
