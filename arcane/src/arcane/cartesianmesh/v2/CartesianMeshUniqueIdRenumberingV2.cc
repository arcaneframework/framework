// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumberingV2.cc                       (C) 2000-2024 */
/*                                                                           */
/* Renumbering of uniqueId() for Cartesian meshes.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/v2/CartesianMeshUniqueIdRenumberingV2.h"

#include "arcane/utils/PlatformUtils.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/ICartesianMeshPatch.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshUniqueIdRenumberingV2::
CartesianMeshUniqueIdRenumberingV2(ICartesianMesh* cmesh, ICartesianMeshGenerationInfo* gen_info)
: TraceAccessor(cmesh->traceMng())
, m_cartesian_mesh(cmesh)
, m_generation_info(gen_info)
{
  if (platform::getEnvironmentVariable("ARCANE_DEBUG_AMR_RENUMBERING") == "1")
    m_is_verbose = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumberingV2::
renumber()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  Int32 dimension = mesh->dimension();
  Int64 cartesian_global_nb_cell = m_generation_info->globalNbCell();
  info() << "Apply UniqueId renumbering to mesh '" << mesh->name() << "'"
         << " global_nb_cell=" << cartesian_global_nb_cell
         << " global_nb_cell_by_dim=" << m_generation_info->globalNbCells();

  VariableCellInt64 cells_new_uid(VariableBuildInfo(mesh, "ArcaneRenumberCellsNewUid"));
  VariableNodeInt64 nodes_new_uid(VariableBuildInfo(mesh, "ArcaneRenumberNodesNewUid"));
  VariableFaceInt64 faces_new_uid(VariableBuildInfo(mesh, "ArcaneRenumberFacesNewUid"));

  cells_new_uid.fill(-1);
  nodes_new_uid.fill(-1);
  faces_new_uid.fill(-1);

  // Marks the entities derived from the Cartesian mesh as level 0
  // They will not be renumbered
  ICartesianMeshPatch* patch0 = m_cartesian_mesh->patch(0);
  ENUMERATE_ (Cell, icell, patch0->cells()) {
    Cell c{ *icell };
    cells_new_uid[icell] = c.uniqueId().asInt64();
    for (Node n : c.nodes())
      nodes_new_uid[n] = n.uniqueId();
    for (Face f : c.faces())
      faces_new_uid[f] = f.uniqueId();
  }

  // For each level 0 mesh, calculate its index (i,j) in the Cartesian mesh

  // For this, we assume that the mesh was created with the 'CartesianMeshGenerator'
  // (or a generator that has the same numbering) and that the uniqueId() of a cell is:
  //   Int64 cell_unique_id = i + j * all_nb_cell_x;
  // with:
  //   all_nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];

  // In 3D:
  //   Int64 cell_unique_id = i + j * all_nb_cell_x + k * (all_nb_cell_x * all_nb_cell_y);
  // with:
  //   all_nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];
  //   all_nb_cell_y = m_generation_info->globalNbCells()[MD_DirY];

  Int64ConstArrayView global_nb_cells_by_direction = m_generation_info->globalNbCells();
  Int64 nb_cell_x = global_nb_cells_by_direction[MD_DirX];
  if (nb_cell_x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", nb_cell_x);

  Int64 nb_cell_y = global_nb_cells_by_direction[MD_DirY];
  if (nb_cell_y <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", nb_cell_y);

  Int64 nb_cell_z = ((dimension == 2) ? 1 : global_nb_cells_by_direction[MD_DirZ]);
  if (nb_cell_z <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", nb_cell_z);

  if (dimension == 2) {
    ENUMERATE_ (Cell, icell, patch0->cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      Int64 coord_i = uid % nb_cell_x;
      Int64 coord_j = uid / nb_cell_x;
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << cell.uniqueId() << " I=" << coord_i
               << " J=" << coord_j << " nb_cell_x=" << nb_cell_x;
      _applyChildrenCell2D(cell, nodes_new_uid, faces_new_uid, cells_new_uid,
                           coord_i, coord_j, nb_cell_x, nb_cell_y,
                           0, 0, 0, 0);
    }
  }

  else if (dimension == 3) {
    ENUMERATE_ (Cell, icell, patch0->cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      Int64 to2d = uid % (nb_cell_x * nb_cell_y);
      Int64 coord_i = to2d % nb_cell_x;
      Int64 coord_j = to2d / nb_cell_x;
      Int64 coord_k = uid / (nb_cell_x * nb_cell_y);
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << cell.uniqueId() << " I=" << coord_i
               << " J=" << coord_j << " K=" << coord_k
               << " nb_cell_x=" << nb_cell_x << " nb_cell_y=" << nb_cell_y;
      _applyChildrenCell3D(cell, nodes_new_uid, faces_new_uid, cells_new_uid,
                           coord_i, coord_j, coord_k,
                           nb_cell_x, nb_cell_y, nb_cell_z,
                           0, 0, 0, 0);
    }
  }

  // TODO: create a class for this.
  //info() << "Change CellFamily";
  //mesh->cellFamily()->notifyItemsUniqueIdChanged();

  _applyFamilyRenumbering(mesh->cellFamily(), cells_new_uid);
  _applyFamilyRenumbering(mesh->nodeFamily(), nodes_new_uid);
  _applyFamilyRenumbering(mesh->faceFamily(), faces_new_uid);
  mesh->checkValidMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumberingV2::
_applyFamilyRenumbering(IItemFamily* family, VariableItemInt64& items_new_uid)
{
  info() << "Change uniqueId() for family=" << family->name();
  items_new_uid.synchronize();
  ENUMERATE_ (Item, iitem, family->allItems()) {
    Item item{ *iitem };
    Int64 current_uid = item.uniqueId();
    Int64 new_uid = items_new_uid[iitem];
    if (new_uid >= 0 && new_uid != current_uid) {
      if (m_is_verbose)
        info() << "Change ItemUID old=" << current_uid << " new=" << new_uid;
      item.mutableItemBase().setUniqueId(new_uid);
    }
  }
  family->notifyItemsUniqueIdChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumberingV2::
_applyChildrenCell2D(Cell cell, VariableNodeInt64& nodes_new_uid, VariableFaceInt64& faces_new_uid,
                     VariableCellInt64& cells_new_uid,
                     Int64 coord_i, Int64 coord_j,
                     Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y,
                     Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder)
{
  // TODO: to be able to adapt to all refinements, instead of 4,
  // it would be necessary to take the max of nbHChildren()
  // TODO : See if it works for pattern != 2.
  const Int32 pattern = 2;

  const Int64 current_level_nb_node_x = current_level_nb_cell_x + 1;
  const Int64 current_level_nb_node_y = current_level_nb_cell_y + 1;

  const Int64 current_level_nb_face_x = current_level_nb_cell_x + 1;

  // // Non-recursive version for cell_adder, node_adder, and face_adder.
  // cell_adder = 0;
  // node_adder = 0;
  // face_adder = 0;
  // const Int64 parent_level_nb_cell_x = current_level_nb_cell_x / pattern;
  // const Int64 parent_level_nb_cell_y = current_level_nb_cell_y / pattern;
  // Int64 level_i_nb_cell_x = parent_level_nb_cell_x;
  // Int64 level_i_nb_cell_y = parent_level_nb_cell_y;
  // for(Int32 i = current_level-1; i >= 0; i--){
  //   face_adder += (level_i_nb_cell_x * level_i_nb_cell_y) * 2 + level_i_nb_cell_x*2 + level_i_nb_cell_y;
  //   cell_adder += level_i_nb_cell_x * level_i_nb_cell_y;
  //   node_adder += (level_i_nb_cell_x + 1) * (level_i_nb_cell_y + 1);
  //   level_i_nb_cell_x /= pattern;
  //   level_i_nb_cell_y /= pattern;
  // }

  // Renumbers the cell.
  {
    Int64 new_uid = (coord_i + coord_j * current_level_nb_cell_x) + cell_adder;
    if (cells_new_uid[cell] < 0) {
      cells_new_uid[cell] = new_uid;
      if (m_is_verbose)
        info() << "APPLY_CELL_CHILD: uid=" << cell.uniqueId() << " I=" << coord_i << " J=" << coord_j
               << " current_level=" << current_level << " new_uid=" << new_uid << " CellAdder=" << cell_adder;
    }
  }

  // Renumbers the nodes of the current cell.
  // Assumes we have 4 nodes
  // WARNING: we cannot easily maintain the order
  // of the uniqueIds() between the old and new numbering.
  // This invalidates the face orientation, which will need to be redone.
  {
    if (cell.nbNode() != 4)
      ARCANE_FATAL("Invalid number of nodes N={0}, expected=4", cell.nbNode());
    std::array<Int64, 4> new_uids;

    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * current_level_nb_node_x);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * current_level_nb_node_x);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * current_level_nb_node_x);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * current_level_nb_node_x);

    for (Integer z = 0; z < 4; ++z) {
      Node node = cell.node(z);
      if (nodes_new_uid[node] < 0) {
        new_uids[z] += node_adder;
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << new_uids[z];
        nodes_new_uid[node] = new_uids[z];
      }
    }
  }

  // Renumbers the faces
  //  |-0--|--2-|
  // 4|   6|   8|
  //  |-5--|-7--|
  // 9|  11|  13|
  //  |-10-|-12-|
  //
  // With this numbering, TOP < LEFT < BOTTOM < RIGHT
  // Aside from the uniqueIds of the first row of faces, all
  // the uniqueIds are contiguous.
  {
    if (cell.nbFace() != 4)
      ARCANE_FATAL("Invalid number of faces N={0}, expected=4", cell.nbFace());
    std::array<Int64, 4> new_uids;

    // TOP
    // - "(current_level_nb_face_x + current_level_nb_cell_x)" :
    //   the number of LEFT BOTTOM RIGHT faces above.
    // - "coord_j * (current_level_nb_face_x + current_level_nb_cell_x)" :
    //   the total number of LEFT BOTTOM RIGHT faces above.
    // - "coord_i * 2"
    //   we advance two by two on the faces of the same "side".
    new_uids[0] = coord_i * 2 + coord_j * (current_level_nb_face_x + current_level_nb_cell_x);

    // BOTTOM
    // For BOTTOM, it is like TOP but with an additional "number of faces above".
    new_uids[2] = new_uids[0] + (current_level_nb_face_x + current_level_nb_cell_x);
    // LEFT
    // For LEFT, it is the UID of BOTTOM - 1.
    new_uids[3] = new_uids[2] - 1;
    // RIGHT
    // For RIGHT, it is the UID of BOTTOM + 1.
    new_uids[1] = new_uids[2] + 1;

    for (Integer z = 0; z < 4; ++z) {
      Face face = cell.face(z);
      if (faces_new_uid[face] < 0) {
        new_uids[z] += face_adder;
        if (m_is_verbose)
          info() << "APPLY_FACE_CHILD: uid=" << face.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << new_uids[z];
        faces_new_uid[face] = new_uids[z];
      }
    }
  }

  // Renumbers the sub-meshes
  // Assumes we have 4 child cells arranged as follows by cells
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  cell_adder += current_level_nb_cell_x * current_level_nb_cell_y;
  node_adder += current_level_nb_node_x * current_level_nb_node_y;
  face_adder += (current_level_nb_cell_x * current_level_nb_cell_y) * 2 + current_level_nb_cell_x * 2 + current_level_nb_cell_y;

  current_level_nb_cell_x *= pattern;
  current_level_nb_cell_y *= pattern;

  current_level += 1;

  coord_i *= pattern;
  coord_j *= pattern;

  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i + icell % pattern;
    Int64 my_coord_j = coord_j + icell / pattern;

    _applyChildrenCell2D(sub_cell, nodes_new_uid, faces_new_uid, cells_new_uid, my_coord_i, my_coord_j,
                         current_level_nb_cell_x, current_level_nb_cell_y, current_level, cell_adder, node_adder, face_adder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumberingV2::
_applyChildrenCell3D(Cell cell, VariableNodeInt64& nodes_new_uid, VariableFaceInt64& faces_new_uid,
                     VariableCellInt64& cells_new_uid,
                     Int64 coord_i, Int64 coord_j, Int64 coord_k,
                     Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int64 current_level_nb_cell_z,
                     Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder)
{
  // TODO: to be able to adapt to all refinements, instead of 8,
  // we should take the max of nbHChildren()
  // TODO: Check if it works for pattern != 2.
  const Int32 pattern = 2;

  const Int64 current_level_nb_node_x = current_level_nb_cell_x + 1;
  const Int64 current_level_nb_node_y = current_level_nb_cell_y + 1;
  const Int64 current_level_nb_node_z = current_level_nb_cell_z + 1;

  const Int64 current_level_nb_face_x = current_level_nb_cell_x + 1;
  const Int64 current_level_nb_face_y = current_level_nb_cell_y + 1;
  const Int64 current_level_nb_face_z = current_level_nb_cell_z + 1;

  // // Version non récursive pour cell_adder, node_adder et face_adder.
  // cell_adder = 0;
  // node_adder = 0;
  // face_adder = 0;
  // const Int64 parent_level_nb_cell_x = current_level_nb_cell_x / pattern;
  // const Int64 parent_level_nb_cell_y = current_level_nb_cell_y / pattern;
  // const Int64 parent_level_nb_cell_z = current_level_nb_cell_z / pattern;
  // Int64 level_i_nb_cell_x = parent_level_nb_cell_x;
  // Int64 level_i_nb_cell_y = parent_level_nb_cell_y;
  // Int64 level_i_nb_cell_z = parent_level_nb_cell_z;
  // for(Int32 i = current_level-1; i >= 0; i--){
  //   face_adder += (level_i_nb_cell_z + 1) * level_i_nb_cell_x * level_i_nb_cell_y
  //               + (level_i_nb_cell_x + 1) * level_i_nb_cell_y * level_i_nb_cell_z
  //               + (level_i_nb_cell_y + 1) * level_i_nb_cell_z * level_i_nb_cell_x;

  //   cell_adder += level_i_nb_cell_x * level_i_nb_cell_y * level_i_nb_cell_z;
  //   node_adder += (level_i_nb_cell_x + 1) * (level_i_nb_cell_y + 1) * (level_i_nb_cell_z + 1);

  //   level_i_nb_cell_x /= pattern;
  //   level_i_nb_cell_y /= pattern;
  //   level_i_nb_cell_z /= pattern;
  // }

  // Renumbers the cell.
  {
    Int64 new_uid = (coord_i + coord_j * current_level_nb_cell_x + coord_k * current_level_nb_cell_x * current_level_nb_cell_y) + cell_adder;
    if (cells_new_uid[cell] < 0) {
      cells_new_uid[cell] = new_uid;
      if (m_is_verbose)
        info() << "APPLY_CELL_CHILD: uid=" << cell.uniqueId() << " I=" << coord_i << " J=" << coord_j << " K=" << coord_k
               << " current_level=" << current_level << " new_uid=" << new_uid << " CellAdder=" << cell_adder;
    }
  }

  // Renumbers the nodes of the current mesh.
  // Assumes we have 8 nodes
  // WARNING: initially, we cannot easily maintain the order
  // of uniqueIds() between the old and new numbering.
  // This invalidates the face orientation, which will need to be redone.
  {
    if (cell.nbNode() != 8)
      ARCANE_FATAL("Invalid number of nodes N={0}, expected=8", cell.nbNode());
    std::array<Int64, 8> new_uids;
    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);

    new_uids[4] = (coord_i + 0) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[5] = (coord_i + 1) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[6] = (coord_i + 1) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    new_uids[7] = (coord_i + 0) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);

    for (Integer z = 0; z < 8; ++z) {
      Node node = cell.node(z);
      if (nodes_new_uid[node] < 0) {
        new_uids[z] += node_adder;
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << new_uids[z];
        nodes_new_uid[node] = new_uids[z];
      }
    }
  }

  // Renumbers the faces
  // This algorithm is not based on the 2D algorithm.
  // The generated UniqueIDs are contiguous.
  // It is also possible to find the UniqueIDs of the faces
  // using the cell position and the mesh size.
  // Furthermore, the order of the face UniqueIDs of a cell is always the
  // same (in Arcane localId notation (cell.face(i)) : 0, 3, 1, 4, 2, 5).
  // The generated UniqueIDs are therefore the same regardless of the decomposition.
  /*
       x               z
    ┌──►          │ ┌──►
    │             │ │
   y▼12   13   14 │y▼ ┌────┬────┐
      │ 26 │ 27 │ │   │ 24 │ 25 │
      └────┴────┘ │   0    4    8
     15   16   17 │
      │ 28 │ 29 │ │   │    │    │
      └────┴────┘ │   2    6   10
   z=0            │              x=0
  - - - - - - - - - - - - - - - - - -
   z=1            │              x=1
     18   19   20 │   ┌────┬────┐
      │ 32 │ 33 │ │   │ 30 │ 31 │
      └────┴────┘ │   1    5    9
     21   22   23 │
      │ 34 │ 35 │ │   │    │    │
      └────┴────┘ │   3    7   11
                  │
  */
  // We have a cube decomposed into eight cells (2x2x2).
  // The diagram above represents the faces of the cells of this cube with
  // the uniqueIDs that the algorithm will generate (without face_adder).
  // For this algorithm, we start with the "xy" faces.
  // We enumerate first in x, then in y, then in z.
  // Once the "xy" faces are numbered, we do the "yz" faces.
  // Always the same numbering order.
  // We finish with the "zx" faces, still in the same order.
  //
  // In the implementation below, we do the numbering
  // cell by cell.
  const Int64 total_face_xy = current_level_nb_face_z * current_level_nb_cell_x * current_level_nb_cell_y;
  const Int64 total_face_xy_yz = total_face_xy + current_level_nb_face_x * current_level_nb_cell_y * current_level_nb_cell_z;
  const Int64 total_face_xy_yz_zx = total_face_xy_yz + current_level_nb_face_y * current_level_nb_cell_z * current_level_nb_cell_x;
  {
    if (cell.nbFace() != 6)
      ARCANE_FATAL("Invalid number of faces N={0}, expected=6", cell.nbFace());
    std::array<Int64, 6> new_uids;

    //// Version originale :
    // new_uids[0] = (coord_k * current_level_nb_cell_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i);

    // new_uids[3] = ((coord_k+1) * current_level_nb_cell_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i);

    // new_uids[1] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_face_x)
    //             + (coord_i) + total_face_xy;

    // new_uids[4] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_face_x)
    //             + (coord_i+1) + total_face_xy;

    // new_uids[2] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i) + total_face_xy_yz;

    // new_uids[5] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y)
    //             + ((coord_j+1) * current_level_nb_cell_x)
    //             + (coord_i) + total_face_xy_yz;
    ////

    const Int64 nb_cell_before_j = coord_j * current_level_nb_cell_x;

    new_uids[0] = (coord_k * current_level_nb_cell_x * current_level_nb_cell_y) + nb_cell_before_j + (coord_i);

    new_uids[3] = new_uids[0] + current_level_nb_cell_x * current_level_nb_cell_y;

    new_uids[1] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y) + (coord_j * current_level_nb_face_x) + (coord_i) + total_face_xy;

    new_uids[4] = new_uids[1] + 1;

    new_uids[2] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y) + nb_cell_before_j + (coord_i) + total_face_xy_yz;

    new_uids[5] = new_uids[2] + current_level_nb_cell_x;

    for (Integer z = 0; z < 6; ++z) {
      Face face = cell.face(z);
      if (faces_new_uid[face] < 0) {
        new_uids[z] += face_adder;
        if (m_is_verbose)
          info() << "APPLY_FACE_CHILD: uid=" << face.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << new_uids[z];
        faces_new_uid[face] = new_uids[z];
      }
    }
  }

  // Renumbers the sub-meshes
  // Assumes we have 8 child cells (2x2x2) arranged as follows by cells
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  cell_adder += current_level_nb_cell_x * current_level_nb_cell_y * current_level_nb_cell_z;
  node_adder += current_level_nb_node_x * current_level_nb_node_y * current_level_nb_node_z;
  face_adder += total_face_xy_yz_zx;

  coord_i *= pattern;
  coord_j *= pattern;
  coord_k *= pattern;

  current_level_nb_cell_x *= pattern;
  current_level_nb_cell_y *= pattern;
  current_level_nb_cell_z *= pattern;

  current_level += 1;

  const Int32 pattern_cube = pattern * pattern;

  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i + icell % pattern;
    Int64 my_coord_j = coord_j + (icell % pattern_cube) / pattern;
    Int64 my_coord_k = coord_k + icell / pattern_cube;

    _applyChildrenCell3D(sub_cell, nodes_new_uid, faces_new_uid, cells_new_uid, my_coord_i, my_coord_j, my_coord_k,
                         current_level_nb_cell_x, current_level_nb_cell_y, current_level_nb_cell_z,
                         current_level, cell_adder, node_adder, face_adder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
