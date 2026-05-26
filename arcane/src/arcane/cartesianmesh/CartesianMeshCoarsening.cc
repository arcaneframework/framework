// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Coarsening of a Cartesian mesh.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshCoarsening.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CartesianGridDimension.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/SimpleSVGMeshExporter.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshStats.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshCoarsening::
CartesianMeshCoarsening(ICartesianMesh* m)
: TraceAccessor(m->traceMng())
, m_cartesian_mesh(m)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL", true))
    m_verbosity_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Returns the max of uniqueId() of entities in a group
Int64 CartesianMeshCoarsening::
_getMaxUniqueId(const ItemGroup& group)
{
  Int64 max_offset = 0;
  ENUMERATE_ (Item, iitem, group) {
    Item item = *iitem;
    if (max_offset < item.uniqueId())
      max_offset = item.uniqueId();
  }
  return max_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening::
createCoarseCells()
{
  if (m_is_create_coarse_called)
    ARCANE_FATAL("This method has already been called");
  m_is_create_coarse_called = true;

  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_patch != 1)
    ARCANE_FATAL("This method is only valid for 1 patch (nb_patch={0})", nb_patch);

  if (!mesh->isAmrActivated())
    ARCANE_FATAL("AMR is not activated for this case");

  // TODO: Delete the ghost cells then reconstruct them
  // TODO: Update the information in CellDirectionMng
  // of ownNbCell(), globalNbCell(), ...

  Integer nb_dir = mesh->dimension();
  if (nb_dir != 2)
    ARCANE_FATAL("This method is only valid for 2D mesh");

  IParallelMng* pm = mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  info() << "CoarseCartesianMesh nb_direction=" << nb_dir;

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    Int32 nb_own_cell = cdm.ownNbCell();
    info() << "NB_OWN_CELL dir=" << idir << " n=" << nb_own_cell;
    if ((nb_own_cell % 2) != 0)
      ARCANE_FATAL("Invalid number of cells ({0}) for direction {1}. Should be a multiple of 2",
                   nb_own_cell, idir);
  }

  // Calculates the offset for creating uniqueIds.
  // We take the max of the uniqueIds of faces and cells as the offset.
  // Eventually, with Cartesian numbering everywhere, we will be able to determine
  // this value directly.
  Int64 max_cell_uid = _getMaxUniqueId(mesh->allCells());
  Int64 max_face_uid = _getMaxUniqueId(mesh->allFaces());
  const Int64 coarse_grid_cell_offset = 1 + pm->reduce(Parallel::ReduceMax, math::max(max_cell_uid, max_face_uid));
  m_first_own_cell_unique_id_offset = coarse_grid_cell_offset;

  CellDirectionMng cdm_x(m_cartesian_mesh->cellDirection(0));
  CellDirectionMng cdm_y(m_cartesian_mesh->cellDirection(1));

  const Int64 global_nb_cell_x = cdm_x.globalNbCell();
  const Int64 global_nb_cell_y = cdm_y.globalNbCell();
  CartesianGridDimension refined_grid_dim(global_nb_cell_x, global_nb_cell_y);
  CartesianGridDimension coarse_grid_dim(global_nb_cell_x / 2, global_nb_cell_y / 2);
  CartesianGridDimension::CellUniqueIdComputer2D refined_cell_uid_computer(refined_grid_dim.getCellComputer2D(0));
  CartesianGridDimension::NodeUniqueIdComputer2D refined_node_uid_computer(refined_grid_dim.getNodeComputer2D(0));
  CartesianGridDimension::CellUniqueIdComputer2D coarse_cell_uid_computer(coarse_grid_dim.getCellComputer2D(coarse_grid_cell_offset));
  CartesianGridDimension::FaceUniqueIdComputer2D coarse_face_uid_computer(coarse_grid_dim.getFaceComputer2D(coarse_grid_cell_offset));

  // For the coarse cells and faces, the nodes already exist
  // Therefore, we cannot use the Cartesian connectivity of the coarse grid
  // for them (we can do this when patch AMR with duplication is active)
  // For now, we use the numbering of the refined grid.

  // TODO: Calculate the number of faces and cells and allocate accordingly.
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> cells_infos;
  Int32 nb_coarse_face = 0;
  Int32 nb_coarse_cell = 0;
  //! List of the first child of each coarse cell
  UniqueArray<Int64> first_child_cell_unique_ids;
  ENUMERATE_ (Cell, icell, mesh->ownCells()) {
    Cell cell = *icell;
    Int64 cell_uid = cell.uniqueId();
    Int64x3 cell_xy = refined_cell_uid_computer.compute(cell_uid);
    const Int64 cell_x = cell_xy.x;
    const Int64 cell_y = cell_xy.y;
    // Since we are coarsening by 2, only take cells whose coordinates
    // are topologically even
    if ((cell_x % 2) != 0 || (cell_y % 2) != 0)
      continue;
    if (is_verbose)
      info() << "CELLCoarse uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y;
    const Int64 coarse_cell_x = cell_x / 2;
    const Int64 coarse_cell_y = cell_y / 2;
    std::array<Int64, 4> node_uids_container;
    ArrayView<Int64> node_uids(node_uids_container);
    node_uids[0] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 0);
    node_uids[1] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 0);
    node_uids[2] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 2);
    node_uids[3] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 2);
    if (is_verbose)
      info() << "CELLNodes uid=" << node_uids;
    std::array<Int64, 4> coarse_face_uids = coarse_face_uid_computer.computeForCell(coarse_cell_x, coarse_cell_y);
    const ItemTypeInfo* cell_type = cell.typeInfo();
    // Adds the 4 faces
    for (Int32 z = 0; z < 4; ++z) {
      ItemTypeInfo::LocalFace lface = cell_type->localFace(z);
      faces_infos.add(IT_Line2);
      faces_infos.add(coarse_face_uids[z]);
      faces_infos.add(node_uids[lface.node(0)]);
      faces_infos.add(node_uids[lface.node(1)]);
      ++nb_coarse_face;
    }
    // Adds the cell
    {
      cells_infos.add(IT_Quad4);
      cells_infos.add(coarse_cell_uid_computer.compute(coarse_cell_x, coarse_cell_y));
      for (Int32 z = 0; z < 4; ++z)
        cells_infos.add(node_uids[z]);
      ++nb_coarse_cell;
      first_child_cell_unique_ids.add(cell.uniqueId());
    }
  }

  UniqueArray<Int32> cells_local_ids;
  UniqueArray<Int32> faces_local_ids;
  cells_local_ids.resize(nb_coarse_cell);
  faces_local_ids.resize(nb_coarse_face);
  mesh->modifier()->addFaces(MeshModifierAddFacesArgs(nb_coarse_face, faces_infos, faces_local_ids));
  mesh->modifier()->addCells(MeshModifierAddCellsArgs(nb_coarse_cell, cells_infos, cells_local_ids));

  // Now that the coarse cells are created, we must indicate
  // that they are parents.
  IItemFamily* cell_family = mesh->cellFamily();

  // Positions the owners of the new cells
  // and adds a flag (ItemFlags::II_UserMark1) to mark them.
  // This will be used to destroy the refined cells later.
  {
    ENUMERATE_ (Cell, icell, cell_family->view(cells_local_ids)) {
      Cell cell = *icell;
      cell.mutableItemBase().setOwner(my_rank, my_rank);
      cell.mutableItemBase().addFlags(ItemFlags::II_UserMark1);
    }
    cell_family->notifyItemsOwnerChanged();
  }

  // We must assign an owner to the faces.
  // Since the new faces use an already existing node, we take the owner
  // of the first node of the face
  {
    IItemFamily* face_family = mesh->faceFamily();
    ENUMERATE_ (Face, iface, face_family->view(faces_local_ids)) {
      Face face = *iface;
      Int32 owner = face.node(0).owner();
      face.mutableItemBase().setOwner(owner, my_rank);
    }
    face_family->notifyItemsOwnerChanged();
  }

  // Updates the mesh
  mesh->modifier()->endUpdate();

  // After calling endUpdate(), the local IDs no longer change.
  // We can use this to keep the list of refined cells for each coarse cell
  m_coarse_cells.resize(nb_coarse_cell);
  m_refined_cells.resize(nb_coarse_cell, 4);

  {
    CellInfoListView cells(mesh->cellFamily());
    UniqueArray<Int32> first_child_cell_local_ids(nb_coarse_cell);
    cell_family->itemsUniqueIdToLocalId(first_child_cell_local_ids, first_child_cell_unique_ids);
    Int32 coarse_index = 0;
    std::array<Int32, 4> sub_cell_lids_container;
    ArrayView<Int32> sub_cell_lids(sub_cell_lids_container);

    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      Cell coarse_cell = *icell;
      if (!(coarse_cell.itemBase().flags() & ItemFlags::II_UserMark1))
        continue;
      // Removes the flag
      coarse_cell.mutableItemBase().removeFlags(ItemFlags::II_UserMark1);
      m_coarse_cells[coarse_index] = coarse_cell.itemLocalId();
      Cell first_child_cell = cells[first_child_cell_local_ids[coarse_index]];
      // Starting from the first sub-cell, we can know the other 3
      // because they are respectively to the right, top-right, and top.
      sub_cell_lids[0] = first_child_cell.localId();
      sub_cell_lids[1] = cdm_x[first_child_cell].next().localId();
      sub_cell_lids[2] = cdm_y[CellLocalId(sub_cell_lids[1])].next().localId();
      sub_cell_lids[3] = cdm_y[first_child_cell].next().localId();
      if (is_verbose)
        info() << "AddChildForCoarseCell i=" << coarse_index << " coarse=" << ItemPrinter(coarse_cell)
               << " children_lid=" << sub_cell_lids;
      for (Int32 z = 0; z < 4; ++z) {
        CellLocalId sub_local_id = CellLocalId(sub_cell_lids[z]);
        m_refined_cells[coarse_index][z] = sub_local_id;
        if (is_verbose)
          info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(cells[sub_local_id]);
      }
      ++coarse_index;
    }
  }

  if (is_verbose) {
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Final cell=" << ItemPrinter(cell) << " level=" << cell.level();
    }
  }

  const bool dump_coarse_mesh = false;
  if (dump_coarse_mesh) {
    String filename = String::format("mesh_coarse_{0}.svg", my_rank);
    std::ofstream ofile(filename.localstr());
    SimpleSVGMeshExporter writer(ofile);
    writer.write(mesh->allCells());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening::
removeRefinedCells()
{
  if (!m_is_create_coarse_called)
    ARCANE_FATAL("You need to call createCoarseCells() before");
  if (m_is_remove_refined_called)
    ARCANE_FATAL("This method has already been called");
  m_is_remove_refined_called = true;

  IMesh* mesh = m_cartesian_mesh->mesh();
  IMeshModifier* mesh_modifier = mesh->modifier();

  // Deletes all refined cells as well as all ghost cells
  {
    std::unordered_set<Int32> coarse_cells_set;
    for (Int32 cell_lid : m_coarse_cells)
      coarse_cells_set.insert(cell_lid);

    UniqueArray<Int32> cells_to_remove;
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      Int32 local_id = icell.itemLocalId();
      Cell cell = *icell;
      if (!cell.isOwn() || (coarse_cells_set.find(local_id) != coarse_cells_set.end()))
        cells_to_remove.add(local_id);
    }
    mesh_modifier->removeCells(cells_to_remove);
    mesh_modifier->endUpdate();
  }

  // Reconstructs the ghost cells
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();

  // Displays the statistics of the new mesh
  {
    MeshStats ms(traceMng(), mesh, mesh->parallelMng());
    ms.dumpStats();
  }

  _recomputeMeshGenerationInfo();

  // We must recalculate the new directions
  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recalculates the information on the number of cells per direction.
 */
void CartesianMeshCoarsening::
_recomputeMeshGenerationInfo()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(mesh, false);
  if (!cmgi)
    return;

  // Coarsening factor
  const Int32 cf = 2;

  {
    ConstArrayView<Int64> v = cmgi->ownCellOffsets();
    cmgi->setOwnCellOffsets(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  {
    ConstArrayView<Int64> v = cmgi->globalNbCells();
    cmgi->setGlobalNbCells(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  {
    ConstArrayView<Int32> v = cmgi->ownNbCells();
    cmgi->setOwnNbCells(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  cmgi->setFirstOwnCellUniqueId(m_first_own_cell_unique_id_offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
