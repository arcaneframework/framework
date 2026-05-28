// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Coarsening of a Cartesian mesh.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/SmallArray.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CartesianGridDimension.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/SimpleSVGMeshExporter.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshStats.h"
#include "arcane/core/IGhostLayerMng.h"

#include "arcane/mesh/CellFamily.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshCoarsening2::
CartesianMeshCoarsening2(ICartesianMesh* m)
: TraceAccessor(m->traceMng())
, m_cartesian_mesh(m)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL", true))
    m_verbosity_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Returns the max of uniqueId() of entities in a group
Int64 CartesianMeshCoarsening2::
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

void CartesianMeshCoarsening2::
_writeMeshSVG(const String& name)
{
  if (m_verbosity_level <= 0)
    return;
  IMesh* mesh = m_cartesian_mesh->mesh();
  if (mesh->dimension() != 2)
    return;
  IParallelMng* pm = mesh->parallelMng();
  const Int32 mesh_rank = pm->commRank();
  String filename = String::format("mesh_{0}_{1}.svg", name, mesh_rank);
  info() << "WriteMesh name=" << filename;
  std::ofstream ofile(filename.localstr());
  SimpleSVGMeshExporter writer(ofile);
  writer.write(mesh->allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Doubles the ghost layer of the initial mesh.
 *
 * This will then allow for the correct ghost layer value for the final coarse mesh.
 */
void CartesianMeshCoarsening2::
_doDoubleGhostLayers()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  IMeshModifier* mesh_modifier = mesh->modifier();
  IGhostLayerMng* gm = mesh->ghostLayerMng();
  // We must at least use version 3 to support
  // multiple ghost layers
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);
  Int32 nb_ghost_layer = gm->nbGhostLayer();
  gm->setNbGhostLayer(nb_ghost_layer * 2);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();
  // Restores the initial number of ghost layers
  gm->setNbGhostLayer(nb_ghost_layer);

  // Since the number of entities has been modified, we must recalculate the directions
  m_cartesian_mesh->computeDirections();

  // Writes the new mesh
  _writeMeshSVG("double_ghost");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
createCoarseCells()
{
  if (m_is_create_coarse_called)
    ARCANE_FATAL("This method has already been called");
  m_is_create_coarse_called = true;

  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  IParallelMng* pm = mesh->parallelMng();
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_patch != 1)
    ARCANE_FATAL("This method is only valid for 1 patch (nb_patch={0})", nb_patch);

  if (!mesh->isAmrActivated())
    ARCANE_FATAL("AMR is not activated for this case");

  Integer nb_dir = mesh->dimension();
  if (nb_dir != 2 && nb_dir != 3)
    ARCANE_FATAL("This method is only valid for 2D or 3D mesh (dim={0})", nb_dir);

  info() << "CoarseCartesianMesh nb_direction=" << nb_dir;

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    Int32 nb_own_cell = cdm.ownNbCell();
    info() << "NB_OWN_CELL dir=" << idir << " n=" << nb_own_cell;
    if ((nb_own_cell % 2) != 0)
      ARCANE_FATAL("Invalid number of cells ({0}) for direction {1}. Should be a multiple of 2",
                   nb_own_cell, idir);
  }

  _writeMeshSVG("orig");

  // Doubles the ghost layer
  _doDoubleGhostLayers();

  if (is_verbose) {
    SmallArray<Int64, 8> uids;
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Orig cell=" << ItemPrinter(cell) << " Face0=" << cell.face(0).uniqueId()
             << " Face1=" << cell.face(1).uniqueId() << " level=" << cell.level();
      Int32 nb_node = cell.nbNode();
      uids.resize(nb_node);
      for (Int32 i = 0; i < nb_node; ++i)
        uids[i] = cell.node(i).uniqueId();
      info() << "Orig cell_uid=" << cell.uniqueId() << " Nodes=" << uids;
      Int32 nb_face = cell.nbFace();
      uids.resize(nb_face);
      for (Int32 i = 0; i < nb_face; ++i)
        uids[i] = cell.face(i).uniqueId();
      info() << "Orig cell_uid=" << cell.uniqueId() << " Faces=" << uids;
    }
  }

  // Calculate the offset for creating uniqueIds().
  // We take the max of the uniqueIds() of faces and cells as the offset.
  // Eventually, with Cartesian numbering everywhere, we will be able to determine
  // this value directly
  Int64 max_cell_uid = _getMaxUniqueId(mesh->allCells());
  Int64 max_face_uid = _getMaxUniqueId(mesh->allFaces());
  const Int64 coarse_grid_cell_offset = 1 + pm->reduce(Parallel::ReduceMax, math::max(max_cell_uid, max_face_uid));
  m_first_own_cell_unique_id_offset = coarse_grid_cell_offset;
  info() << "FirstCellUniqueIdOffset=" << m_first_own_cell_unique_id_offset;
  if (nb_dir == 3)
    _createCoarseCells3D();
  else if (nb_dir == 2)
    _createCoarseCells2D();
  else
    ARCANE_FATAL("Invalid dimension '{0}'", nb_dir);

  mesh->modifier()->endUpdate();

  if (is_verbose) {
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Final cell=" << ItemPrinter(cell) << " Face0=" << cell.face(0).uniqueId()
             << " Face1=" << cell.face(1).uniqueId() << " level=" << cell.level();
    }
  }

  _recomputeMeshGenerationInfo();

  // Displays the statistics of the new mesh
  {
    MeshStats ms(traceMng(), mesh, mesh->parallelMng());
    ms.dumpStats();
  }

  //! Creates the patch with the child cells
  {
    CellGroup parent_cells = mesh->allLevelCells(0);
    m_cartesian_mesh->_internalApi()->addPatchFromExistingChildren(parent_cells.view().localIds());
  }

  // Recalculates the synchronization information
  // This is not necessary for AMR because this information will be recalculated
  // during refinement, but since we don't know if we will refine
  // afterwards, it is better to calculate this information in all cases.
  mesh->computeSynchronizeInfos();

  // We must recalculate the new directions after the modifications
  // and the addition of the patch.
  m_cartesian_mesh->computeDirections();

  _writeMeshSVG("coarse");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
_createCoarseCells2D()
{
  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  IParallelMng* pm = mesh->parallelMng();
  const Int32 my_rank = pm->commRank();

  CellDirectionMng cdm_x(m_cartesian_mesh->cellDirection(0));
  CellDirectionMng cdm_y(m_cartesian_mesh->cellDirection(1));

  const Int64 global_nb_cell_x = cdm_x.globalNbCell();
  const Int64 global_nb_cell_y = cdm_y.globalNbCell();
  CartesianGridDimension refined_grid_dim(global_nb_cell_x, global_nb_cell_y);
  CartesianGridDimension coarse_grid_dim(global_nb_cell_x / 2, global_nb_cell_y / 2);
  CartesianGridDimension::CellUniqueIdComputer2D refined_cell_uid_computer(refined_grid_dim.getCellComputer2D(0));
  CartesianGridDimension::NodeUniqueIdComputer2D refined_node_uid_computer(refined_grid_dim.getNodeComputer2D(0));
  CartesianGridDimension::CellUniqueIdComputer2D coarse_cell_uid_computer(coarse_grid_dim.getCellComputer2D(m_first_own_cell_unique_id_offset));
  CartesianGridDimension::FaceUniqueIdComputer2D coarse_face_uid_computer(coarse_grid_dim.getFaceComputer2D(m_first_own_cell_unique_id_offset));

  // For the coarse cells and faces, the nodes already exist
  // Therefore, we cannot use the Cartesian connectivity of the coarse grid
  // for them (we can do this when patch-based AMR with duplication is active)
  // For now, we use the numbering of the refined grid.

  // TODO: Calculate the number of faces and cells in advance and allocate accordingly.
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> cells_infos;
  Int32 nb_coarse_face = 0;
  Int32 nb_coarse_cell = 0;
  //! List of the first child of each coarse cell
  UniqueArray<Cell> first_child_cells;

  UniqueArray<Int64> refined_cells_lids;
  UniqueArray<Int32> coarse_cells_owner;
  UniqueArray<Int32> coarse_faces_owner;

  ENUMERATE_ (Cell, icell, mesh->allCells()) {
    Cell cell = *icell;
    Int64 cell_uid = cell.uniqueId();
    Int64x3 cell_xy = refined_cell_uid_computer.compute(cell_uid);
    const Int64 cell_x = cell_xy.x;
    const Int64 cell_y = cell_xy.y;
    // Necessary for recalculating ghost cells. We consider these
    // cells as if they were just refined.
    cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined);
    // Since we coarsen by 2, we only take cells whose topological coordinates
    // are even
    if ((cell_x % 2) != 0 || (cell_y % 2) != 0)
      continue;
    if (is_verbose)
      info() << "CellToCoarse refined_uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y;
    coarse_cells_owner.add(cell.owner());
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
      Int64 node_uid0 = node_uids[lface.node(0)];
      Int64 node_uid1 = node_uids[lface.node(1)];

      if (node_uid0 > node_uid1)
        std::swap(node_uid0, node_uid1);
      if (is_verbose)
        info() << "ADD_FACE coarse_uid=" << coarse_face_uids[z] << " n0=" << node_uid0 << " n1=" << node_uid1;
      faces_infos.add(node_uid0);
      faces_infos.add(node_uid1);
      ++nb_coarse_face;
    }
    // Adds the cell
    {
      cells_infos.add(IT_Quad4);
      Int64 coarse_cell_uid = coarse_cell_uid_computer.compute(coarse_cell_x, coarse_cell_y);
      cells_infos.add(coarse_cell_uid);
      if (is_verbose)
        info() << "CoarseCellUid=" << coarse_cell_uid;
      m_coarse_cells_uid.add(coarse_cell_uid);
      for (Int32 z = 0; z < 4; ++z)
        cells_infos.add(node_uids[z]);
      ++nb_coarse_cell;
      first_child_cells.add(cell);
    }
    // From the first sub-cell, we can know the other 3
    // because they are respectively to the right, upper right, and upper.
    {
      std::array<Int32, 4> sub_cell_lids_container;
      ArrayView<Int32> sub_lids(sub_cell_lids_container);
      Cell cell1 = cdm_x[cell].next();
      // Checks the validity of the sub-cells.
      // Normally there should be no problems unless the
      // number of cells in each direction of the sub-domain
      // is not an even number.
      if (cell1.null())
        ARCANE_FATAL("Bad right cell for cell {0}", ItemPrinter(cell));
      Cell cell2 = cdm_y[cell1].next();
      if (cell2.null())
        ARCANE_FATAL("Bad upper right cell for cell {0}", ItemPrinter(cell));
      Cell cell3 = cdm_y[cell].next();
      if (cell3.null())
        ARCANE_FATAL("Bad upper cell for cell {0}", ItemPrinter(cell));
      sub_lids[0] = cell.localId();
      sub_lids[1] = cell1.localId();
      sub_lids[2] = cell2.localId();
      sub_lids[3] = cell3.localId();
      // We must assign an owner to the faces.
      // These new faces will have the same owner as the refined faces
      // they correspond to
      //info() << "CELL_NB_FACE=" << cell.nbFace() << " " << cell1.nbFace() << " " << cell2.nbFace() << " " << cell3.nbFace();
      coarse_faces_owner.add(cell.face(0).owner());
      coarse_faces_owner.add(cell1.face(1).owner());
      coarse_faces_owner.add(cell2.face(2).owner());
      coarse_faces_owner.add(cell3.face(3).owner());
      for (Int32 i = 0; i < 4; ++i)
        refined_cells_lids.add(sub_lids[i]);
    }
  }

  // Builds the faces
  UniqueArray<Int32> faces_local_ids(nb_coarse_face);
  mesh->modifier()->addFaces(nb_coarse_face, faces_infos, faces_local_ids);

  // Builds the cells
  // Indicates that we are not allowed to build the faces on the fly.
  // Normally they have all been added via addFaces();
  UniqueArray<Int32> cells_local_ids(nb_coarse_cell);
  MeshModifierAddCellsArgs add_cells_args(nb_coarse_cell, cells_infos, cells_local_ids);
  add_cells_args.setAllowBuildFaces(false);
  mesh->modifier()->addCells(add_cells_args);

  IItemFamily* cell_family = mesh->cellFamily();

  // Now that the coarse meshes are created, we must indicate
  // that they are parent cells.
  using mesh::CellFamily;
  CellInfoListView cells(mesh->cellFamily());
  CellFamily* true_cell_family = ARCANE_CHECK_POINTER(dynamic_cast<CellFamily*>(cell_family));
  std::array<Int32, 4> sub_cell_lids_container;
  ArrayView<Int32> sub_cell_lids(sub_cell_lids_container);
  for (Int32 i = 0; i < nb_coarse_cell; ++i) {
    Int32 coarse_cell_lid = cells_local_ids[i];
    Cell coarse_cell = cells[coarse_cell_lid];
    Cell first_child_cell = first_child_cells[i];
    // Starting from the first sub-mesh, we can know the other 3
    // because they are respectively to the right, upper right, and upper.
    sub_cell_lids[0] = first_child_cell.localId();
    sub_cell_lids[1] = cdm_x[first_child_cell].next().localId();
    sub_cell_lids[2] = cdm_y[CellLocalId(sub_cell_lids[1])].next().localId();
    sub_cell_lids[3] = cdm_y[first_child_cell].next().localId();
    if (is_verbose)
      info() << "AddChildForCoarseCell i=" << i << " coarse=" << ItemPrinter(coarse_cell)
             << " children_lid=" << sub_cell_lids;
    for (Int32 z = 0; z < 4; ++z) {
      Cell child_cell = cells[sub_cell_lids[z]];
      if (is_verbose)
        info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(child_cell);
      true_cell_family->_addParentCellToCell(child_cell, coarse_cell);
    }
    true_cell_family->_addChildrenCellsToCell(coarse_cell, sub_cell_lids);
  }

  // Positions the owners of the new meshes and faces
  {
    IItemFamily* face_family = mesh->faceFamily();
    Int32 index = 0;
    ENUMERATE_ (Cell, icell, cell_family->view(cells_local_ids)) {
      Cell cell = *icell;
      Int32 owner = coarse_cells_owner[index];
      cell.mutableItemBase().setOwner(owner, my_rank);
      const Int64 sub_cell_index = index * 4;
      for (Int32 z = 0; z < 4; ++z) {
        cell.face(z).mutableItemBase().setOwner(coarse_faces_owner[sub_cell_index + z], my_rank);
      }
      ++index;
    }
    cell_family->notifyItemsOwnerChanged();
    face_family->notifyItemsOwnerChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
_createCoarseCells3D()
{
  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  IParallelMng* pm = mesh->parallelMng();
  const Int32 my_rank = pm->commRank();

  CellDirectionMng cdm_x(m_cartesian_mesh->cellDirection(0));
  CellDirectionMng cdm_y(m_cartesian_mesh->cellDirection(1));
  CellDirectionMng cdm_z(m_cartesian_mesh->cellDirection(2));

  const Int64 global_nb_cell_x = cdm_x.globalNbCell();
  const Int64 global_nb_cell_y = cdm_y.globalNbCell();
  const Int64 global_nb_cell_z = cdm_z.globalNbCell();
  CartesianGridDimension refined_grid_dim(global_nb_cell_x, global_nb_cell_y, global_nb_cell_z);
  CartesianGridDimension coarse_grid_dim(global_nb_cell_x / 2, global_nb_cell_y / 2, global_nb_cell_z / 2);
  CartesianGridDimension::CellUniqueIdComputer3D refined_cell_uid_computer(refined_grid_dim.getCellComputer3D(0));
  CartesianGridDimension::NodeUniqueIdComputer3D refined_node_uid_computer(refined_grid_dim.getNodeComputer3D(0));
  CartesianGridDimension::CellUniqueIdComputer3D coarse_cell_uid_computer(coarse_grid_dim.getCellComputer3D(m_first_own_cell_unique_id_offset));
  CartesianGridDimension::FaceUniqueIdComputer3D coarse_face_uid_computer(coarse_grid_dim.getFaceComputer3D(m_first_own_cell_unique_id_offset));

  // For the coarse meshes and faces, the nodes already exist
  // Therefore, we cannot use the Cartesian connectivity of the coarse grid
  // for them (we can do that when AMR by patch with duplication is active)
  // For now, we use the numbering of the refined grid.

  // TODO: Calculate the number of faces and meshes in advance and allocate accordingly.
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> cells_infos;
  Int32 nb_coarse_face = 0;
  Int32 nb_coarse_cell = 0;
  //! List of the first child of each coarse mesh
  UniqueArray<Cell> first_child_cells;

  UniqueArray<Int64> refined_cells_lids;
  UniqueArray<Int32> coarse_cells_owner;
  UniqueArray<Int32> coarse_faces_owner;

  static constexpr Int32 const_cell_nb_node = 8;
  static constexpr Int32 const_cell_nb_face = 6;
  static constexpr Int32 const_cell_nb_sub_cell = 8;
  static constexpr Int32 const_face_nb_node = 4;

  // List of uniqueId() of nodes of created faces
  SmallArray<Int64, const_face_nb_node> face_node_uids(const_face_nb_node);
  // Ordered list of nodes of created faces
  SmallArray<Int64, const_face_nb_node> face_sorted_node_uids(const_face_nb_node);
  ENUMERATE_ (Cell, icell, mesh->allCells()) {
    Cell cell = *icell;
    Int64 cell_uid = cell.uniqueId();
    Int64x3 cell_xyz = refined_cell_uid_computer.compute(cell_uid);
    const Int64 cell_x = cell_xyz.x;
    const Int64 cell_y = cell_xyz.y;
    const Int64 cell_z = cell_xyz.z;
    // Necessary for ghost mesh recalculation. We consider these
    // meshes as if they have just been refined.
    cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined);
    // Since we refine by 2, only take meshes whose topological coordinates
    // are even
    if ((cell_x % 2) != 0 || (cell_y % 2) != 0 || (cell_z % 2) != 0)
      continue;
    if (is_verbose)
      info() << "CellToCoarse refined_uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y << " z=" << cell_z;
    coarse_cells_owner.add(cell.owner());
    const Int64 coarse_cell_x = cell_x / 2;
    const Int64 coarse_cell_y = cell_y / 2;
    const Int64 coarse_cell_z = cell_z / 2;
    std::array<Int64, const_cell_nb_node> node_uids_container;
    ArrayView<Int64> node_uids(node_uids_container);
    node_uids[0] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 0, cell_z + 0);
    node_uids[1] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 0, cell_z + 0);
    node_uids[2] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 2, cell_z + 0);
    node_uids[3] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 2, cell_z + 0);
    node_uids[4] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 0, cell_z + 2);
    node_uids[5] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 0, cell_z + 2);
    node_uids[6] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 2, cell_z + 2);
    node_uids[7] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 2, cell_z + 2);
    if (is_verbose)
      info() << "CELLNodes uid=" << node_uids;
    std::array<Int64, const_cell_nb_face> coarse_face_uids = coarse_face_uid_computer.computeForCell(coarse_cell_x, coarse_cell_y, coarse_cell_z);
    const ItemTypeInfo* cell_type = cell.typeInfo();

    // Add the 6 faces
    for (Int32 z = 0; z < const_cell_nb_face; ++z) {
      ItemTypeInfo::LocalFace lface = cell_type->localFace(z);
      faces_infos.add(IT_Quad4);
      faces_infos.add(coarse_face_uids[z]);
      for (Int32 knode = 0; knode < const_face_nb_node; ++knode)
        face_node_uids[knode] = node_uids[lface.node(knode)];
      MeshUtils::reorderNodesOfFace(face_node_uids, face_sorted_node_uids);
      if (is_verbose)
        info() << "ADD_FACE coarse_uid=" << coarse_face_uids[z] << " n=" << face_sorted_node_uids;
      faces_infos.addRange(face_sorted_node_uids);
      ++nb_coarse_face;
    }

    // Add the mesh
    {
      cells_infos.add(IT_Hexaedron8);
      Int64 coarse_cell_uid = coarse_cell_uid_computer.compute(coarse_cell_x, coarse_cell_y, coarse_cell_z);
      cells_infos.add(coarse_cell_uid);
      if (is_verbose)
        info() << "CoarseCellUid=" << coarse_cell_uid;
      m_coarse_cells_uid.add(coarse_cell_uid);
      for (Int32 z = 0; z < const_cell_nb_node; ++z)
        cells_infos.add(node_uids[z]);
      ++nb_coarse_cell;
      first_child_cells.add(cell);
    }

    // Starting from the first sub-mesh, we can know the other 7
    // because they are respectively to the right, upper right, and upper,
    // above, above right, above upper right and .
    {
      std::array<Int32, const_cell_nb_sub_cell> sub_cell_lids_container;
      ArrayView<Int32> sub_lids(sub_cell_lids_container);
      Cell cell1 = cdm_x[cell].next();
      // Checks the validity of the sub-meshes.
      // Normally there should be no problems unless the
      // number of meshes in each direction of the sub-domain
      // is not an even number.
      if (cell1.null())
        ARCANE_FATAL("Bad right cell for cell {0}", ItemPrinter(cell));
      Cell cell2 = cdm_y[cell1].next();
      if (cell2.null())
        ARCANE_FATAL("Bad upper right cell for cell {0}", ItemPrinter(cell));
      Cell cell3 = cdm_y[cell].next();
      if (cell3.null())
        ARCANE_FATAL("Bad upper cell for cell {0}", ItemPrinter(cell));

      Cell cell4 = cdm_z[cell].next();
      if (cell4.null())
        ARCANE_FATAL("Bad top cell for cell {0}", ItemPrinter(cell));

      Cell cell5 = cdm_x[cell4].next();
      if (cell5.null())
        ARCANE_FATAL("Bad top right cell for cell {0}", ItemPrinter(cell));
      Cell cell6 = cdm_y[cell5].next();
      if (cell6.null())
        ARCANE_FATAL("Bad top upper right cell for cell {0}", ItemPrinter(cell));
      Cell cell7 = cdm_y[cell4].next();
      if (cell7.null())
        ARCANE_FATAL("Bad top upper cell for cell {0}", ItemPrinter(cell));

      sub_lids[0] = cell.localId();
      sub_lids[1] = cell1.localId();
      sub_lids[2] = cell2.localId();
      sub_lids[3] = cell3.localId();
      sub_lids[4] = cell4.localId();
      sub_lids[5] = cell5.localId();
      sub_lids[6] = cell6.localId();
      sub_lids[7] = cell7.localId();
      // We need to assign an owner to the faces.
      // These new faces will have the same owner as the refined faces
      // they correspond to
      //info() << "CELL_NB_FACE=" << cell.nbFace() << " " << cell1.nbFace() << " " << cell2.nbFace() << " " << cell3.nbFace();
      coarse_faces_owner.add(cell.face(0).owner());
      coarse_faces_owner.add(cell1.face(1).owner());
      coarse_faces_owner.add(cell2.face(2).owner());
      coarse_faces_owner.add(cell3.face(3).owner());

      coarse_faces_owner.add(cell4.face(4).owner());
      coarse_faces_owner.add(cell5.face(5).owner());

      for (Int32 i = 0; i < const_cell_nb_sub_cell; ++i)
        refined_cells_lids.add(sub_lids[i]);
    }
  }

  // Constructs the faces
  UniqueArray<Int32> faces_local_ids(nb_coarse_face);
  mesh->modifier()->addFaces(nb_coarse_face, faces_infos, faces_local_ids);

  // Constructs the meshes
  // Indicates that we do not have the right to build faces on the fly.
  // Normally they have all been added via addFaces();
  UniqueArray<Int32> cells_local_ids(nb_coarse_cell);
  MeshModifierAddCellsArgs add_cells_args(nb_coarse_cell, cells_infos, cells_local_ids);
  add_cells_args.setAllowBuildFaces(false);
  mesh->modifier()->addCells(add_cells_args);

  IItemFamily* cell_family = mesh->cellFamily();

  // Now that the coarse meshes are created, we must indicate
  // that they are parent cells.
  using mesh::CellFamily;
  CellInfoListView cells(mesh->cellFamily());
  CellFamily* true_cell_family = ARCANE_CHECK_POINTER(dynamic_cast<CellFamily*>(cell_family));
  std::array<Int32, const_cell_nb_sub_cell> sub_cell_lids_container;
  ArrayView<Int32> sub_cell_lids(sub_cell_lids_container);
  for (Int32 i = 0; i < nb_coarse_cell; ++i) {
    Int32 coarse_cell_lid = cells_local_ids[i];
    Cell coarse_cell = cells[coarse_cell_lid];
    Cell first_child_cell = first_child_cells[i];
    // Starting from the first sub-mesh, we can know the other 3
    // because they are respectively to the right, upper right, and upper.
    sub_cell_lids[0] = first_child_cell.localId();
    sub_cell_lids[1] = cdm_x[first_child_cell].next().localId();
    sub_cell_lids[2] = cdm_y[CellLocalId(sub_cell_lids[1])].next().localId();
    sub_cell_lids[3] = cdm_y[first_child_cell].next().localId();

    Cell top_first_child_cell = cdm_z[first_child_cell].next();
    sub_cell_lids[4] = top_first_child_cell.localId();
    sub_cell_lids[5] = cdm_x[top_first_child_cell].next().localId();
    sub_cell_lids[6] = cdm_y[CellLocalId(sub_cell_lids[5])].next().localId();
    sub_cell_lids[7] = cdm_y[top_first_child_cell].next().localId();

    if (is_verbose)
      info() << "AddChildForCoarseCell i=" << i << " coarse=" << ItemPrinter(coarse_cell)
             << " children_lid=" << sub_cell_lids;
    for (Int32 z = 0; z < const_cell_nb_sub_cell; ++z) {
      Cell child_cell = cells[sub_cell_lids[z]];
      if (is_verbose)
        info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(child_cell);
      true_cell_family->_addParentCellToCell(child_cell, coarse_cell);
    }
    true_cell_family->_addChildrenCellsToCell(coarse_cell, sub_cell_lids);
  }

  // Positions the owners of the new meshes and faces
  {
    IItemFamily* face_family = mesh->faceFamily();
    Int32 index = 0;
    ENUMERATE_ (Cell, icell, cell_family->view(cells_local_ids)) {
      Cell cell = *icell;
      Int32 owner = coarse_cells_owner[index];
      cell.mutableItemBase().setOwner(owner, my_rank);
      const Int64 sub_cell_index = index * const_cell_nb_face;
      for (Int32 z = 0; z < const_cell_nb_face; ++z) {
        cell.face(z).mutableItemBase().setOwner(coarse_faces_owner[sub_cell_index + z], my_rank);
      }
      ++index;
    }
    cell_family->notifyItemsOwnerChanged();
    face_family->notifyItemsOwnerChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recalculates the information about the number of meshes per direction.
 */
void CartesianMeshCoarsening2::
_recomputeMeshGenerationInfo()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(mesh, false);
  if (!cmgi)
    return;

  // Refinement coefficient
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

void CartesianMeshCoarsening2::
removeRefinedCells()
{
  if (!m_is_create_coarse_called)
    ARCANE_FATAL("You need to call createCoarseCells() before");
  if (m_is_remove_refined_called)
    ARCANE_FATAL("This method has already been called");
  m_is_remove_refined_called = true;

  const bool is_verbose = m_verbosity_level > 0;

  IMesh* mesh = m_cartesian_mesh->mesh();
  IMeshModifier* mesh_modifier = mesh->modifier();

  info() << "RemoveRefinedCells nb_coarse_cell=" << m_coarse_cells_uid.size();
  if (is_verbose)
    info() << "CoarseCells=" << m_coarse_cells_uid;

  // Remove all refined meshes as well as all ghost meshes
  {
    std::unordered_set<Int64> coarse_cells_set;
    for (Int64 cell_uid : m_coarse_cells_uid)
      coarse_cells_set.insert(cell_uid);
    UniqueArray<Int32> cells_to_remove;
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Int32 local_id = icell.itemLocalId();
      Cell cell = *icell;
      if (coarse_cells_set.find(cell.uniqueId()) == coarse_cells_set.end())
        cells_to_remove.add(local_id);
    }
    if (is_verbose)
      info() << "CellsToRemove n=" << cells_to_remove.size() << " list=" << cells_to_remove;
    mesh_modifier->removeCells(cells_to_remove);
    mesh_modifier->endUpdate();
  }

  // Reconstruct ghost meshes
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();

  // Display statistics of the new mesh
  {
    MeshStats ms(traceMng(), mesh, mesh->parallelMng());
    ms.dumpStats();
  }

  // We must recalculate the new directions
  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
