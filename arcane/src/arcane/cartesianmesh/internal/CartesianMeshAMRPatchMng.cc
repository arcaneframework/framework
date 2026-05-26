// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.cc                                 (C) 2000-2026 */
/*                                                                           */
/* AMR Patch Manager for a Cartesian Mesh.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianMeshAMRPatchMng.h"

#include "arcane/utils/Array2View.h"
#include "arcane/utils/Array3View.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector2.h"
#include "arcane/utils/Vector3.h"

#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/materials/IMeshMaterialMng.h"

#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/cartesianmesh/internal/CartesianPatchGroup.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAMRPatchMng::
CartesianMeshAMRPatchMng(ICartesianMesh* cmesh, ICartesianMeshNumberingMngInternal* numbering_mng)
: TraceAccessor(cmesh->mesh()->traceMng())
, m_mesh(cmesh->mesh())
, m_cmesh(cmesh)
, m_num_mng(numbering_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * For the comments of this method, we consider the following coordinate system:
 *       (top)
 *         y    (front)
 *         ^    z
 *         |   /
 *         | /
 *  (left) ------->x (right)
 *  (rear)(bottom)
 */
void CartesianMeshAMRPatchMng::
refine()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 max_level = 0;

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_internals.add(cell);
      if (cell.level() > max_level)
        max_level = cell.level();
    }
  }
  m_num_mng->prepareLevel(max_level + 1);

  UniqueArray<Int64> cells_infos;
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;

  UniqueArray<Int64> node_uid_change_owner_only;
  UniqueArray<Int64> face_uid_change_owner_only;

  // Two arrays allowing the retrieval of unique IDs of nodes and faces
  // from each child mesh upon calling getNodeUids()/getFaceUids().
  UniqueArray<Int64> child_nodes_uids(m_num_mng->nbNodeByCell());
  UniqueArray<Int64> child_faces_uids(m_num_mng->nbFaceByCell());

  // We must record the parent meshes of each child mesh to update connectivities
  // when creating the meshes.
  UniqueArray<Int32> parent_cells;

  // Maps replacing ghost meshes.
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_flags;

  {
    // We only need these two flags for surrounding meshes.
    // (II_Refine to know if surrounding meshes are in the same patch)
    // (II_Inactive to know if surrounding meshes are already refined)
    Int32 useful_flags = ItemFlags::II_Refine + ItemFlags::II_Inactive;
    _shareInfosOfCellsAroundPatch(cell_to_refine_internals, around_parent_cells_uid_to_owner, around_parent_cells_uid_to_flags, useful_flags);
  }

  if (m_mesh->dimension() == 2) {

    // Masks for "child neighbors" and "parent neighbors in the same patch" cases.
    // These masks determine whether a node should be created based on
    // the surrounding meshes.
    // For example, if we are studying a child mesh and there is
    // a child mesh to the left, we should not create nodes 0 and 3 (mask_node_if_cell_left[]) (because
    // they have already been created by the mesh on the left).
    // The same applies to neighboring parent meshes: if we are on a child mesh located
    // on the left side of the parent mesh (child meshes 0 and 2 in the case of a
    // refinement pattern = 2), and there is a parent mesh to the left and that parent mesh
    // is currently ((being refined and in our subdomain) or (is inactive)), we apply
    // the mask_node_if_cell_left[] rule because the nodes were created by it and we want to avoid
    // duplicate nodes.
    // These masks also allow us to determine the owner of the nodes in
    // the case of multiple subdomains.
    // For example, if we are on a child mesh located
    // on the left side of the parent mesh (child meshes 0 and 2 in the case of a
    // refinement pattern = 2), and there is a parent mesh to the left and that mesh
    // (belongs to another subdomain) and (is currently being refined),
    // we create this node but assign the process that owns
    // the parent mesh on the left as the owner.
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true };

    constexpr bool mask_node_if_cell_right[] = { true, false, false, true };
    constexpr bool mask_node_if_cell_top[] = { true, true, false, false };

    constexpr bool mask_face_if_cell_left[] = { true, true, true, false };
    constexpr bool mask_face_if_cell_bottom[] = { false, true, true, true };

    constexpr bool mask_face_if_cell_right[] = { true, false, true, true };
    constexpr bool mask_face_if_cell_top[] = { true, true, false, true };

    // For sizing:
    // - we have "cell_to_refine_internals.size() * 4" child meshes,
    // - for each mesh, we have 2 pieces of info (mesh type and mesh uniqueId)
    // - for each mesh, we have "m_num_mng->getNbNode()" uniqueIds (the uniqueIds of each node in the mesh).
    cells_infos.reserve((cell_to_refine_internals.size() * 4) * (2 + m_num_mng->nbNodeByCell()));

    // For sizing, maximum:
    // - we have "cell_to_refine_internals.size() * 12" faces
    // - for each face, we have 2 pieces of info (face type and face uniqueId)
    // - for each face, we have 2 node uniqueIds.
    faces_infos.reserve((cell_to_refine_internals.size() * 12) * (2 + 2));

    // For sizing, maximum:
    // - we have (cell_to_refine_internals.size() * 9) node uniqueIds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 9);

    FixedArray<Int64, 9> uid_cells_around_parent_cell_1d;
    FixedArray<Int32, 9> owner_cells_around_parent_cell_1d;
    FixedArray<Int32, 9> flags_cells_around_parent_cell_1d;

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();
      const bool parent_cell_is_own = (parent_cell.owner() == my_rank);

      const CartCoord parent_coord_x = m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, parent_cell_level);
      const CartCoord parent_coord_y = m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, parent_cell_level);

      const CartCoord child_coord_x = m_num_mng->offsetLevelToLevel(parent_coord_x, parent_cell_level, parent_cell_level + 1);
      const CartCoord child_coord_y = m_num_mng->offsetLevelToLevel(parent_coord_y, parent_cell_level, parent_cell_level + 1);

      const Int32 pattern = m_num_mng->pattern();

      m_num_mng->cellUniqueIdsAroundCell(parent_cell, uid_cells_around_parent_cell_1d.view());

      for (Int32 i = 0; i < 9; ++i) {
        const Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // If uid_cell != -1, there might be a mesh (but we don't know if it is actually present).
        // If around_parent_cells_uid_to_owner[uid_cell] != -1, there is indeed a mesh.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else {
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // To simplify, we use 2D views. (array[Y][X]).
      ConstArray2View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3);

      // #priority_owner_2d
      // Here are the priorities for node and face ownership:
      // ┌─────────┐
      // │6   7   8│
      // └───────┐ │
      // ┌─┐ ┌─┐ │ │
      // │3│ │4│ │5│
      // │ │ └─┘ └─┘
      // │ └───────┐
      // │0   1   2│
      // └─────────┘
      //
      // ^y
      // |
      // ->x

      // #arcane_order_to_around_2d
      // Note for 2D Cartesian meshes:
      // The face iterators iterate in the order (for mesh 4 here):
      //  0. Face between [4, 1],
      //  1. Face between [4, 5],
      //  2. Face between [4, 7],
      //  3. Face between [4, 3],
      //
      // The node iterators iterate in the order (for mesh 4 here):
      //  0. Node between [4, 0]
      //  1. Node between [4, 2]
      //  2. Node between [4, 8]
      //  3. Node between [4, 6]

      // Each number designates a parent mesh and a priority (0 being the highest priority).
      // 4 = parent_cell ("us")

      // Example 1:
      // We are looking to refine level 0 meshes (i.e., create level 1 meshes).
      // At the bottom, there are no meshes.
      // On the left (priority 3), there is a mesh that is already refined (flag "II_Inactive").
      // We are priority 4, so we are prioritized. Therefore, the nodes and faces we share
      // belong to it.

      // Example 2:
      // We are looking to refine level 0 meshes (i.e., create level 1 meshes).
      // At the top, there are already refined meshes (flag "II_Inactive").
      // We are prioritized over them, so we recover the ownership of the nodes and faces we share. This change of ownership must be signaled to them.

      // We simplify using a boolean array.
      // If true, we must apply the ownership priority.
      // If false, we consider that there is no mesh at the defined position.
      FixedArray<FixedArray<bool, 3>, 3> is_cell_around_parent_cell_present_and_useful;

      // For meshes that prioritize us, we must look at both flags.
      // If a mesh has the "II_Refine" flag, we do not exist for it, so it takes ownership
      // of the faces and nodes we share.
      // If a mesh has the "II_Inactive" flag, it already has the correct owners.
      // In any case, if true, the faces and nodes we share belong to them.
      is_cell_around_parent_cell_present_and_useful[0][0] = ((uid_cells_around_parent_cell(0, 0) != -1) && (flags_cells_around_parent_cell(0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1] = ((uid_cells_around_parent_cell(0, 1) != -1) && (flags_cells_around_parent_cell(0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2] = ((uid_cells_around_parent_cell(0, 2) != -1) && (flags_cells_around_parent_cell(0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][0] = ((uid_cells_around_parent_cell(1, 0) != -1) && (flags_cells_around_parent_cell(1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      // is_cell_around_parent_cell_present_and_useful[1][1] = parent_cell;

      // For non-prioritized meshes, we must look at only one flag.
      // If a mesh has the "II_Inactive" flag, it must be notified that we are taking ownership
      // of the nodes and faces we share.
      // We do not look at the "II_Refine" flag because, if these meshes are also being refined,
      // they know that we exist and that we obtain the ownership of the nodes and faces we share.
      // In summary, if true, the faces and nodes we share belong to us.
      is_cell_around_parent_cell_present_and_useful[1][2] = ((uid_cells_around_parent_cell(1, 2) != -1) && (flags_cells_around_parent_cell(1, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[2][0] = ((uid_cells_around_parent_cell(2, 0) != -1) && (flags_cells_around_parent_cell(2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1] = ((uid_cells_around_parent_cell(2, 1) != -1) && (flags_cells_around_parent_cell(2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2] = ((uid_cells_around_parent_cell(2, 2) != -1) && (flags_cells_around_parent_cell(2, 2) & ItemFlags::II_Inactive));

      // In addition to checking if each parent mesh around our parent mesh exists and possesses (II_Inactive) or will possess (II_Refine) children...
      // ... we check if each parent mesh is present in our subdomain, whether it is a ghost mesh or not.
      auto is_cell_around_parent_cell_in_subdomain = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (flags_cells_around_parent_cell(y, x) & ItemFlags::II_UserMark1);
      };

      // ... we check if each parent mesh is owned by the same owner as our parent mesh.
      auto is_cell_around_parent_cell_same_owner = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) == owner_cells_around_parent_cell(1, 1));
      };

      // ... we check if each parent mesh has a different owner compared to our parent mesh.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };

      // We iterate over all child meshes.
      for (CartCoord j = child_coord_y; j < child_coord_y + pattern; ++j) {
        for (CartCoord i = child_coord_x; i < child_coord_x + pattern; ++i) {
          parent_cells.add(parent_cell.localId());
          total_nb_cells++;

          const Int64 child_cell_uid = m_num_mng->cellUniqueId(CartCoord2(i, j), parent_cell_level + 1);
          // debug() << "Child -- x : " << i << " -- y : " << j << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

          m_num_mng->cellNodeUniqueIds(CartCoord2(i, j), parent_cell_level + 1, child_nodes_uids);
          m_num_mng->cellFaceUniqueIds(CartCoord2(i, j), parent_cell_level + 1, child_faces_uids);

          constexpr Integer type_cell = IT_Quad4;
          constexpr Integer type_face = IT_Line2;

          // Cell Part.
          cells_infos.add(type_cell);
          cells_infos.add(child_cell_uid);
          for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
            cells_infos.add(child_nodes_uids[nc]);
          }

          // Face Part.
          for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
            Integer child_face_owner = -1;
            bool is_new_face = false;

            // Two parts:
            // First, we check if we should create face l. To do this, we must check if it is present on the
            // adjacent mesh.
            // For left/bottom, the principle is the same. If the child mesh is entirely to the left/bottom of the parent mesh, we check
            // if there is a parent mesh to the left/bottom. Otherwise, we create the face. If yes, we check the mask to know if we
            // should create the face.
            // For right/top, the principle is different from left/bottom. We only follow the mask if we are entirely to the right/top
            // of the parent mesh. Otherwise, we always create the right/top faces.
            // Finally, we use the "is_cell_around_parent_cell_in_subdomain" array. If the adjacent parent mesh is in
            // our subdomain, the faces shared with our parent mesh may already exist; in this case,
            // there is no duplicate.
            if (
            ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_face_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_face_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_face_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_face_if_cell_top[l])) {
              is_new_face = true;
              faces_infos.add(type_face);
              faces_infos.add(child_faces_uids[l]);

              // The face nodes are always nodes l and l+1
              // because we use the same exploration for both cases.
              for (Integer nc = l; nc < l + 2; nc++) {
                faces_infos.add(child_nodes_uids[nc % m_num_mng->nbNodeByCell()]);
              }
              total_nb_faces++;

              // By default, parent_cell is the owner of the new face.
              child_face_owner = owner_cells_around_parent_cell(1, 1);
            }

            // Second part.
            // We must now find the correct owner for the face. Aside from the "is_cell_around_parent_cell_same_owner" array,
            // the condition is identical to the one above.
            // The change of array is important because from here on, we are sure that the face we are interested in exists.
            // The new array allows us to know if the adjacent mesh is also ours or not. If not, then
            // an ownership change is possible, according to the priorities defined above. We do not need to know
            // if the mesh is present in the subdomain.
            if (
            ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_face_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_face_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_face_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_face_if_cell_top[l])) {
              // Here, the condition construction is the same every time.
              // The first boolean (i == child_coord_x) checks if the child is on the correct side of the parent mesh.
              // The second boolean (!mask_face_if_cell_left[l]) tells us if face l is indeed
              // the shared face with the adjacent parent mesh.
              // The third boolean (is_cell_around_parent_cell_different_owner(1, 0)) checks if there is an
              // adjacent mesh that takes ownership of the face or to whom we take ownership.

              // Furthermore, there are two different cases depending on the priorities defined above:
              // - either we are not prioritized, so we assign the priority owner to our face,
              // - or we are prioritized, so we position ourselves as the owner of the face and must notify
              //   all other processes (the former owner process as well as processes that may
              //   have the face as a ghost).

              // Finally, in the case of ownership change, only the process taking over ownership must
              // communicate about it. Processes that only possess the ghost face should not
              // communicate (but they can locally define the correct owner, TODO Possible Optimization?).

              // On the left, priority 3 < 4, so it takes ownership of the face.
              if (i == child_coord_x && (!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_different_owner(1, 0)) {
                child_face_owner = owner_cells_around_parent_cell(1, 0);
              }

              // At the bottom, priority 1 < 4, so it takes ownership of the face.
              else if (j == child_coord_y && (!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(0, 1)) {
                child_face_owner = owner_cells_around_parent_cell(0, 1);
              }

              // Otherwise, parent_cell is the owner of the face.
              else {

                // Otherwise, it is an internal face belonging to parent_cell.
                child_face_owner = owner_cells_around_parent_cell(1, 1);
              }
            }

            // If there is a face creation and/or an ownership change.
            if (child_face_owner != -1) {
              face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

              // When there is an ownership change without face creation,
              // we must set aside the uniqueIds of these faces to be able to iterate over them later.
              if (!is_new_face) {
                face_uid_change_owner_only.add(child_faces_uids[l]);
                // debug() << "Child face (change owner) -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- face : " << l
                //         << " -- uid_face : " << child_faces_uids[l]
                //         << " -- owner : " << child_face_owner;
              }
              else {
                // debug() << "Child face (create face)  -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- face : " << l
                //         << " -- uid_face : " << child_faces_uids[l]
                //         << " -- owner : " << child_face_owner;
              }
            }
          }

          // Node Part.
          // This part is quite similar to the face part, except that there can be
          // more possible owners.
          for (Int32 l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
            Int32 child_node_owner = -1;
            bool is_new_node = false;

            // Two parts:
            // First, we check if we should create node l. To do this, we must check if it is present on the
            // adjacent mesh.
            // For left/bottom, the principle is the same. If the child mesh is entirely to the left/bottom of the parent mesh, we check
            // if there is a parent mesh to the left/bottom. Otherwise, we create the node. If yes, we check the mask to know if we
            // should create the node.
            // For right/top, the principle is different from left/bottom. We only follow the mask if the child mesh is entirely to the right/top
            // of the parent mesh. Otherwise, we always create the right/top nodes.
            // Finally, we use the "is_cell_around_parent_cell_in_subdomain" array. If the adjacent parent mesh is in
            // our subdomain, the nodes shared with our parent mesh may already exist; in this case,
            // there is no duplicate.
            if (
            ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_node_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_node_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_node_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_node_if_cell_top[l])) {
              is_new_node = true;
              nodes_infos.add(child_nodes_uids[l]);
              total_nb_nodes++;

              // By default, parent_cell is the owner of the new node.
              child_node_owner = owner_cells_around_parent_cell(1, 1);
            }

            // Second part.
            // We must now find the correct owner for the node. Aside from the array "is_cell_around_parent_cell_same_owner",
            // the condition is identical to the one above.
            // The change of array is important because from here, we are sure that the node we are interested in exists.
            // The new array allows us to know if the neighboring cell is also ours or not. If not, then
            // an owner change is possible, according to the priorities defined above. We do not need to know
            // if the cell is present in the subdomain.
            if (
            ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_node_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_node_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_node_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_node_if_cell_top[l])) {
              // Compared to faces that only have two possible owners, a node can
              // have up to four.
              // (And yes, in 3D, it's even more fun!)

              // If the node is on the left side of the parent cell ("on the left face").
              if (i == child_coord_x && (!mask_node_if_cell_left[l])) {

                // If the node is on the bottom of the parent cell ("on the bottom face").
                // So, node in bottom left (same position as the parent cell node).
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                  // Priority 0 < 4.
                  if (is_cell_around_parent_cell_different_owner(0, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 0);
                  }

                  // Priority 1 < 4.
                  else if (is_cell_around_parent_cell_different_owner(0, 1)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 1);
                  }

                  // Priority 3 < 4.
                  else if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // If the node is on the top of the parent cell ("on the top face").
                // So, node in top left (same position as the parent cell node).
                else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                  // Priority 3 < 4.
                  if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  // Otherwise, parent_cell is the owner of the node.
                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // If the node is somewhere on the parent left face...
                else {
                  // If there is a cell to the left, it is the owner of the node.
                  if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  // Otherwise, parent_cell is the owner of the node.
                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }
              }

              // If the node is on the right side of the parent cell ("on the right face").
              else if (i == (child_coord_x + pattern - 1) && (!mask_node_if_cell_right[l])) {

                // If the node is on the bottom of the parent cell ("on the bottom face").
                // So, node in bottom right (same position as the parent cell node).
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                  // Priority 1 < 4.
                  if (is_cell_around_parent_cell_different_owner(0, 1)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 1);
                  }

                  // Priority 2 < 4.
                  else if (is_cell_around_parent_cell_different_owner(0, 2)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 2);
                  }

                  // Otherwise, parent_cell is the owner of the node.
                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // If the node is on the top of the parent cell ("on the top face").
                // So, node in top right (same position as the parent cell node).
                else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // If the node is somewhere on the parent right face...
                else {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }
              }

              // If the node is neither on the parent left face nor on the parent right face...
              else {

                // If the node is on the bottom of the parent cell ("on the bottom face") and
                // there is a bottom cell with priority 1 < 4, it is the owner of the node.
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(0, 1)) {
                  child_node_owner = owner_cells_around_parent_cell(0, 1);
                }

                // If the node is on the top of the parent cell ("on the top face") and
                // there is a top cell with priority 7 > 4, parent_cell is the owner of the node.
                else if (parent_cell_is_own && j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l]) && is_cell_around_parent_cell_different_owner(2, 1)) {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // Nodes that are not on any face of the parent cell.
                else {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }
              }
            }

            // If there is a node creation and/or an owner change.
            if (child_node_owner != -1) {
              node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

              // When there is an owner change without node creation,
              // we must set aside the uniqueIds of these nodes to be able to
              // iterate over them later.
              if (!is_new_node) {
                node_uid_change_owner_only.add(child_nodes_uids[l]);
                // debug() << "Child node (change owner) -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- node : " << l
                //         << " -- uid_node : " << child_nodes_uids[l]
                //         << " -- owner : " << child_node_owner;
              }
              else {
                // debug() << "Child node (create node)  -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- node : " << l
                //         << " -- uid_node : " << child_nodes_uids[l]
                //         << " -- owner : " << child_node_owner;
              }
            }
          }
        }
      }
    }
  }

  // For 3D, it is very similar, just a bit longer. I am copying the comments, but with some adaptations.
  else if (m_mesh->dimension() == 3) {

    // Masks for "child neighbors" and "parent neighbors of the same patch" cases.
    // These masks allow us to know whether we should create a node or not depending on
    // the surrounding cells.
    // For example, if we are studying a child cell and there is
    // a child cell to the left, we should not create nodes 0, 3, 4, 7 (mask_node_if_cell_left[]) (because
    // they have already been created by the cell to the left).
    // Same for neighboring parent cells: if we are on a child cell located
    // on the left side of the parent cell (child cells 0, 2, 4, 6 in the case of a
    // refinement pattern = 2), there is a parent cell to the left and that parent cell
    // is currently ((being refined and in our subdomain) or (is inactive)), we apply
    // the mask_node_if_cell_left[] rule because the nodes were created by it and we want to avoid
    // duplicate nodes.
    // These masks also allow us to determine the owner of the nodes in
    // the case of multiple subdomains.
    // For example, if we are on a child cell located
    // on the left side of the parent cell (child cells 0, 2, 4, 6 in the case of a
    // refinement pattern = 2), there is a parent cell to the left and that cell
    // (belongs to another subdomain) and (is being refined),
    // we create this node but we give it as owner the process to which belongs
    // the parent cell to the left.
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false, false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true, false, false, true, true };
    constexpr bool mask_node_if_cell_rear[] = { false, false, false, false, true, true, true, true };

    constexpr bool mask_node_if_cell_right[] = { true, false, false, true, true, false, false, true };
    constexpr bool mask_node_if_cell_top[] = { true, true, false, false, true, true, false, false };
    constexpr bool mask_node_if_cell_front[] = { true, true, true, true, false, false, false, false };

    constexpr bool mask_face_if_cell_left[] = { true, false, true, true, true, true };
    constexpr bool mask_face_if_cell_bottom[] = { true, true, false, true, true, true };
    constexpr bool mask_face_if_cell_rear[] = { false, true, true, true, true, true };

    constexpr bool mask_face_if_cell_right[] = { true, true, true, true, false, true };
    constexpr bool mask_face_if_cell_top[] = { true, true, true, true, true, false };
    constexpr bool mask_face_if_cell_front[] = { true, true, true, false, true, true };

    // Small difference compared to 2D. For 2D, the position of the face nodes
    // in the "child_nodes_uids" array is always the same (l and l+1, see 2D).
    // For 3D, this is not the case, so we have arrays to have a correspondence
    // between the nodes of each face and the position of the nodes in the "child_nodes_uids" array.
    // (Example: for face 1 (same enumeration order as Arcane), we must take the
    // "nodes_in_face_1" array and therefore the nodes "child_nodes_uids[0]", "child_nodes_uids[3]",
    // "child_nodes_uids[7]" and "child_nodes_uids[4]").
    constexpr Int32 nodes_in_face_0[] = { 0, 1, 2, 3 };
    constexpr Int32 nodes_in_face_1[] = { 0, 3, 7, 4 };
    constexpr Int32 nodes_in_face_2[] = { 0, 1, 5, 4 };
    constexpr Int32 nodes_in_face_3[] = { 4, 5, 6, 7 };
    constexpr Int32 nodes_in_face_4[] = { 1, 2, 6, 5 };
    constexpr Int32 nodes_in_face_5[] = { 3, 2, 6, 7 };

    constexpr Int32 nb_nodes_in_face = 4;

    // For the size:
    // - we have "cell_to_refine_internals.size() * 8" child cells,
    // - for each cell, we have 2 pieces of information (cell type and cell uniqueId)
    // - for each cell, we have "m_num_mng->getNbNode()" uniqueIds (the uniqueIds of each node of the cell).
    cells_infos.reserve((cell_to_refine_internals.size() * 8) * (2 + m_num_mng->nbNodeByCell()));

    // For the size, maximum:
    // - we have "cell_to_refine_internals.size() * 36" child faces,
    // - for each face, we have 2 pieces of information (face type and face uniqueId)
    // - for each face, we have 4 node uniqueIds.
    faces_infos.reserve((cell_to_refine_internals.size() * 36) * (2 + 4));

    // For the size, maximum:
    // - we have (cell_to_refine_internals.size() * 27) node uniqueIds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 27);

    FixedArray<Int64, 27> uid_cells_around_parent_cell_1d;
    FixedArray<Int32, 27> owner_cells_around_parent_cell_1d;
    FixedArray<Int32, 27> flags_cells_around_parent_cell_1d;

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();

      const CartCoord3 parent_coord = m_num_mng->cellUniqueIdToCoord(parent_cell_uid, parent_cell_level);
      const CartCoord3 child_coord = m_num_mng->offsetLevelToLevel(parent_coord, parent_cell_level, parent_cell_level + 1);

      const Int32 pattern = m_num_mng->pattern();

      m_num_mng->cellUniqueIdsAroundCell(parent_cell, uid_cells_around_parent_cell_1d.view());

      for (Integer i = 0; i < 27; ++i) {
        Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // If uid_cell != -1, there might be a cell (but we don't know if it is actually present).
        // If around_parent_cells_uid_to_owner[uid_cell] != -1, there is indeed a cell.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else {
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // To simplify, we use 3D views. (array[Z][Y][X]).
      ConstArray3View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3, 3);

      // #priority_owner_3d
      // Here are the priorities for node and face ownership:
      // ┌──────────┐ │ ┌──────────┐ │ ┌──────────┐
      // │ 6   7   8│ │ │15  16  17│ │ │24  25  26│
      // │          │ │ └───────┐  │ │ │          │
      // │          │ │ ┌──┐┌──┐│  │ │ │          │
      // │ 3   4   5│ │ │12││13││14│ │ │21  22  23│
      // │          │ │ │  │└──┘└──┘ │ │          │
      // │          │ │ │  └───────┐ │ │          │
      // │ 0   1   2│ │ │ 9  10  11│ │ │18  19  20│
      // └──────────┘ │ └──────────┘ │ └──────────┘
      // Z=0          │ Z=1          │ Z=2
      // ("rear")  │              │ ("front")
      //
      // ^y
      // |
      // ->x

      // #arcane_order_to_around_3d
      // Note for 3D Cartesian meshes:
      // The face iterators iterate in the order (for cell 13 here):
      //  0. Face between [13, 4],
      //  1. Face between [13, 12],
      //  2. Face between [13, 10],
      //  3. Face between [13, 22],
      //  4. Face between [13, 14],
      //  5. Face between [13, 16],
      //
      // The node iterators iterate in the order (for cell 13 here):
      //  0. Node between [13, 0]
      //  1. Node between [13, 2]
      //  2. Node between [13, 8]
      //  3. Node between [13, 6]
      //  4. Node between [13, 18]
      //  5. Node between [13, 20]
      //  6. Node between [13, 26]
      //  7. Node between [13, 24]

      // Each number designates a parent cell and a priority (0 being the highest priority).
      // 13 = parent_cell ("us")

      // Example 1:
      // We are looking to refine level 0 cells (thus creating level 1 cells).
      // At the bottom, there are no cells.
      // To the left (thus priority 12), there is a cell that is already refined (flag "II_Inactive").
      // We are priority 13, so we are prioritized. Therefore, the nodes and faces we share
      // belong to it.

      // Example 2:
      // We are looking to refine level 0 cells (thus creating level 1 cells).
      // At the top, there are already refined cells (flag "II_Inactive").
      // We are prioritized over them, so we retrieve the ownership of the nodes and faces we share
      // in common. This ownership change must be signaled to them.

      // We simplify with a boolean array.
      // If true, then we must apply the ownership priority.
      // If false, then we consider that there is no cell at the defined position.
      FixedArray<FixedArray<FixedArray<bool, 3>, 3>, 3> is_cell_around_parent_cell_present_and_useful;

      // For cells that are prioritized over us, we must look at both flags.
      // If a cell has the "II_Refine" flag, we do not exist for it, so it takes ownership
      // of the faces and nodes we share.
      // If a cell has the "II_Inactive" flag, it already has the correct owners.
      // In any case, if true, then the faces and nodes we share belong to them.
      is_cell_around_parent_cell_present_and_useful[0][0][0] = ((uid_cells_around_parent_cell(0, 0, 0) != -1) && (flags_cells_around_parent_cell(0, 0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][0][1] = ((uid_cells_around_parent_cell(0, 0, 1) != -1) && (flags_cells_around_parent_cell(0, 0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][0][2] = ((uid_cells_around_parent_cell(0, 0, 2) != -1) && (flags_cells_around_parent_cell(0, 0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][0] = ((uid_cells_around_parent_cell(0, 1, 0) != -1) && (flags_cells_around_parent_cell(0, 1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][1] = ((uid_cells_around_parent_cell(0, 1, 1) != -1) && (flags_cells_around_parent_cell(0, 1, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][2] = ((uid_cells_around_parent_cell(0, 1, 2) != -1) && (flags_cells_around_parent_cell(0, 1, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][0] = ((uid_cells_around_parent_cell(0, 2, 0) != -1) && (flags_cells_around_parent_cell(0, 2, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][1] = ((uid_cells_around_parent_cell(0, 2, 1) != -1) && (flags_cells_around_parent_cell(0, 2, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][2] = ((uid_cells_around_parent_cell(0, 2, 2) != -1) && (flags_cells_around_parent_cell(0, 2, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][0][0] = ((uid_cells_around_parent_cell(1, 0, 0) != -1) && (flags_cells_around_parent_cell(1, 0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[1][0][1] = ((uid_cells_around_parent_cell(1, 0, 1) != -1) && (flags_cells_around_parent_cell(1, 0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[1][0][2] = ((uid_cells_around_parent_cell(1, 0, 2) != -1) && (flags_cells_around_parent_cell(1, 0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][1][0] = ((uid_cells_around_parent_cell(1, 1, 0) != -1) && (flags_cells_around_parent_cell(1, 1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      // is_cell_around_parent_cell_present_and_useful[1][1][1] = parent_cell;

      // For non-prioritized cells, we only need to look at one flag.
      // If a cell has the "II_Inactive" flag, it must be notified that we are taking ownership
      // of the nodes and faces we share.
      // We do not look at the "II_Refine" flag because, if these cells are also being refined,
      // they know that we exist and that we obtain ownership of the nodes and faces we share.
      // In summary, if true, then the faces and nodes we share belong to us.
      is_cell_around_parent_cell_present_and_useful[1][1][2] = ((uid_cells_around_parent_cell(1, 1, 2) != -1) && (flags_cells_around_parent_cell(1, 1, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[1][2][0] = ((uid_cells_around_parent_cell(1, 2, 0) != -1) && (flags_cells_around_parent_cell(1, 2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[1][2][1] = ((uid_cells_around_parent_cell(1, 2, 1) != -1) && (flags_cells_around_parent_cell(1, 2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[1][2][2] = ((uid_cells_around_parent_cell(1, 2, 2) != -1) && (flags_cells_around_parent_cell(1, 2, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[2][0][0] = ((uid_cells_around_parent_cell(2, 0, 0) != -1) && (flags_cells_around_parent_cell(2, 0, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][0][1] = ((uid_cells_around_parent_cell(2, 0, 1) != -1) && (flags_cells_around_parent_cell(2, 0, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][0][2] = ((uid_cells_around_parent_cell(2, 0, 2) != -1) && (flags_cells_around_parent_cell(2, 0, 2) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][0] = ((uid_cells_around_parent_cell(2, 1, 0) != -1) && (flags_cells_around_parent_cell(2, 1, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][1] = ((uid_cells_around_parent_cell(2, 1, 1) != -1) && (flags_cells_around_parent_cell(2, 1, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][2] = ((uid_cells_around_parent_cell(2, 1, 2) != -1) && (flags_cells_around_parent_cell(2, 1, 2) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][0] = ((uid_cells_around_parent_cell(2, 2, 0) != -1) && (flags_cells_around_parent_cell(2, 2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][1] = ((uid_cells_around_parent_cell(2, 2, 1) != -1) && (flags_cells_around_parent_cell(2, 2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][2] = ((uid_cells_around_parent_cell(2, 2, 2) != -1) && (flags_cells_around_parent_cell(2, 2, 2) & ItemFlags::II_Inactive));

      // In addition to looking at whether each surrounding parent cell exists and possesses (II_Inactive) or will possess (II_Refine) children...
      // ... we look at whether each surrounding parent cell is present in our subdomain, whether it is a ghost cell or not.
      auto is_cell_around_parent_cell_in_subdomain = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (flags_cells_around_parent_cell(z, y, x) & ItemFlags::II_UserMark1);
      };

      // ... we look at whether each surrounding parent cell is owned by the same owner as our parent cell.
      auto is_cell_around_parent_cell_same_owner = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) == owner_cells_around_parent_cell(1, 1, 1));
      };

      // ... we look at whether each surrounding parent cell has a different owner compared to our parent cell.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };

      // We iterate over all child cells.
      for (CartCoord k = child_coord.z; k < child_coord.z + pattern; ++k) {
        for (CartCoord j = child_coord.y; j < child_coord.y + pattern; ++j) {
          for (CartCoord i = child_coord.x; i < child_coord.x + pattern; ++i) {
            parent_cells.add(parent_cell.localId());
            total_nb_cells++;

            const Int64 child_cell_uid = m_num_mng->cellUniqueId(CartCoord3(i, j, k), parent_cell_level + 1);
            // debug() << "Child -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

            m_num_mng->cellNodeUniqueIds(CartCoord3(i, j, k), parent_cell_level + 1, child_nodes_uids);
            m_num_mng->cellFaceUniqueIds(CartCoord3(i, j, k), parent_cell_level + 1, child_faces_uids);

            constexpr Int64 type_cell = IT_Hexaedron8;
            constexpr Int64 type_face = IT_Quad4;

            // Cell part.
            cells_infos.add(type_cell);
            cells_infos.add(child_cell_uid);
            for (Int32 nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
              cells_infos.add(child_nodes_uids[nc]);
            }

            // Face part.
            for (Int32 l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
              Int32 child_face_owner = -1;
              bool is_new_face = false;

              // Two parts:
              // First, we check if we should create face l. To do this, we must check if it is present on the
              // neighboring cell.
              // For left/bottom/rear, the principle is the same. If the child cell is to the far left/bottom/rear of the parent cell, we check
              // if there is a parent cell to the left/bottom/rear. Otherwise, we create the face. If yes, we check the mask to know if we
              // should create the face.
              // For right/top/front, the principle is different from left/bottom/rear. We only follow the mask if we are to the far right/top/front
              // of the parent cell. Otherwise, we always create the right/top/front faces.
              // Finally, we use the "is_cell_around_parent_cell_in_subdomain" array. If the neighboring parent cell is in
              // our subdomain, then the faces shared with our parent cell may already exist, in which case, no duplicate.
              if (
              ((i == child_coord.x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_face_if_cell_left[l]) &&
              ((i != (child_coord.x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_face_if_cell_right[l]) &&
              ((j == child_coord.y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_face_if_cell_bottom[l]) &&
              ((j != (child_coord.y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_face_if_cell_top[l]) &&
              ((k == child_coord.z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_face_if_cell_rear[l]) &&
              ((k != (child_coord.z + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_face_if_cell_front[l])) {
                is_new_face = true;
                faces_infos.add(type_face);
                faces_infos.add(child_faces_uids[l]);

                // We retrieve the position of the face nodes in the "ua_node_uid" array.
                ConstArrayView<Int32> nodes_in_face_l;
                switch (l) {
                case 0:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_0, nb_nodes_in_face);
                  break;
                case 1:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_1, nb_nodes_in_face);
                  break;
                case 2:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_2, nb_nodes_in_face);
                  break;
                case 3:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_3, nb_nodes_in_face);
                  break;
                case 4:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_4, nb_nodes_in_face);
                  break;
                case 5:
                  nodes_in_face_l = ConstArrayView<Int32>::create(nodes_in_face_5, nb_nodes_in_face);
                  break;
                default:
                  ARCANE_FATAL("Bizarre...");
                }
                for (Integer nc : nodes_in_face_l) {
                  faces_infos.add(child_nodes_uids[nc]);
                }
                total_nb_faces++;

                // By default, parent_cell is the owner of the new face.
                child_face_owner = owner_cells_around_parent_cell(1, 1, 1);
              }

              // Second part.
              // We must now find the correct owner for the face. Aside from the "is_cell_around_parent_cell_same_owner" array,
              // the condition is identical to the one above.
              // The change of array is important because from here, we are sure that the face we are interested in exists.
              // The new array allows us to know if the neighboring cell is also ours or not. If not, then
              // an owner change is possible, according to the priorities defined above. We do not need to know
              // if the cell is present in the subdomain.
              if (
              ((i == child_coord.x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_face_if_cell_left[l]) &&
              ((i != (child_coord.x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_face_if_cell_right[l]) &&
              ((j == child_coord.y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_face_if_cell_bottom[l]) &&
              ((j != (child_coord.y + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_face_if_cell_top[l]) &&
              ((k == child_coord.z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_face_if_cell_rear[l]) &&
              ((k != (child_coord.z + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_face_if_cell_front[l])) {
                // Here, the construction of the conditions is the same every time.
                // The first boolean (i == child_coord_x) checks if the child is located
                // on the correct side of the parent cell.
                // The second boolean (!mask_face_if_cell_left[l]) tells us if face l is indeed
                // the shared face with the neighboring parent cell.
                // The third boolean (is_cell_around_parent_cell_different_owner(1, 0)) checks if there is a
                // neighboring cell that takes ownership of the face or to whom we take ownership.

                // Furthermore, there are two different cases depending on the priorities defined above:
                // - either we are not the priority, so we assign the priority owner to our face,
                // - or we are the priority, so we position ourselves as the owner of the face and we must notify
                //   all other processes (the former owner process as well as processes that might
                //   have the ghost face).

                // Finally, in the case of owner change, only the process (re)taking ownership must
                // make a communication about this. Processes only possessing the ghost face must not
                // make a communication (but they can locally define the correct owner, TODO Possible Optimization?).

                // To the left, priority 12 < 13 so it takes ownership of the face.
                if (i == child_coord.x && (!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 0);
                }

                // At the bottom, priority 10 < 13 so it takes ownership of the face.
                else if (j == child_coord.y && (!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                  child_face_owner = owner_cells_around_parent_cell(1, 0, 1);
                }

                // At the rear, priority 4 < 13 so it takes ownership of the face.
                else if (k == child_coord.z && (!mask_face_if_cell_rear[l]) && is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                  child_face_owner = owner_cells_around_parent_cell(0, 1, 1);
                }

                // Otherwise, parent_cell is the owner of the face.
                else {

                  // Otherwise, it is an internal face, so it belongs to parent_cell.
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }

              // If there is a face creation and/or an owner change.
              if (child_face_owner != -1) {
                face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

                // When there is an owner change without face creation,
                // we must set aside the uniqueIds of these faces to be able to
                // iterate over them later.
                if (!is_new_face) {
                  face_uid_change_owner_only.add(child_faces_uids[l]);
                  // debug() << "Child face (change owner) -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- face : " << l
                  //         << " -- uid_face : " << child_faces_uids[l]
                  //         << " -- owner : " << child_face_owner;
                }
                else {
                  // debug() << "Child face (create face)  -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- face : " << l
                  //         << " -- uid_face : " << child_faces_uids[l]
                  //         << " -- owner : " << child_face_owner;
                }
              }
            }

            // Node part.
            // This part is quite similar to the face part, except that there can be
            // more possible owners.
            for (Int32 l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
              Int32 child_node_owner = -1;
              bool is_new_node = false;

              // Two parts:
              // First, we check if we must create node l. To do this, we must check if it is present on the
              // neighboring cell.
              // For left/bottom/rear, the principle is the same. If the child cell is entirely to the left/bottom/rear of the parent cell, we check
              // if there is a parent cell to the left/bottom/rear. Otherwise, we create the node. If yes, we check the mask to know if we
              // must create the node.
              // For right/top/front, the principle is different from left/bottom/rear. We only follow the mask if the child cell is entirely to the right/top/front
              // of the parent cell. Otherwise, we always create the right/top/front nodes.
              // Finally, we use the "is_cell_around_parent_cell_in_subdomain" array. If the neighboring parent cell is in
              // our subdomain, then the nodes common with our parent cell may already exist; in this case,
              // no duplicate.
              if (
              ((i == child_coord.x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_node_if_cell_left[l]) &&
              ((i != (child_coord.x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_node_if_cell_right[l]) &&
              ((j == child_coord.y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_node_if_cell_bottom[l]) &&
              ((j != (child_coord.y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_node_if_cell_top[l]) &&
              ((k == child_coord.z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_node_if_cell_rear[l]) &&
              ((k != (child_coord.z + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_node_if_cell_front[l])) {
                is_new_node = true;
                nodes_infos.add(child_nodes_uids[l]);
                total_nb_nodes++;

                // By default, parent_cell is the owner of the new node.
                child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
              }

              // Second part.
              // We must now find the correct owner for the node. Aside from the "is_cell_around_parent_cell_same_owner" array,
              // the condition is identical to the one above.
              // The change of array is important because from here, we are sure that the node we are interested in exists.
              // The new array allows us to know if the neighboring cell is also ours or not. If not, then
              // an owner change is possible, according to the priorities defined above. We do not need to know
              // if the cell is present in the subdomain.
              if (
              ((i == child_coord.x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_node_if_cell_left[l]) &&
              ((i != (child_coord.x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_node_if_cell_right[l]) &&
              ((j == child_coord.y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_node_if_cell_bottom[l]) &&
              ((j != (child_coord.y + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_node_if_cell_top[l]) &&
              ((k == child_coord.z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_node_if_cell_rear[l]) &&
              ((k != (child_coord.z + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_node_if_cell_front[l])) {

                // Compared to faces that only have two possible owners, a node can
                // have up to eight.

                // If the node is on the left face of the parent cell.
                if (i == child_coord.x && (!mask_node_if_cell_left[l])) {

                  // If the node is on the bottom face of the parent cell.
                  // So the node is on the left-bottom edge.
                  if (j == child_coord.y && (!mask_node_if_cell_bottom[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So the node is on the left, bottom, and rear (same position as the parent cell node).
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 0 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 0);
                      }

                      // Priority 1 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priority 3 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priority 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 9 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priority 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // No cells around.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So the node is on the left, bottom, and front (same position as the parent cell node).
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priority 9 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priority 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Otherwise the node is somewhere on the left-bottom edge...
                    else {

                      // Priority 9 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priority 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // No cells around.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // If the node is on the top face of the parent cell.
                  // So the node is on the left-top edge.
                  else if (j == (child_coord.y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So the node is on the left, top, and rear (same position as the parent cell node).
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 3 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priority 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 6 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 0);
                      }

                      // Priority 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Priority 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So the node is on the left, top, and front (same position as the parent cell node).
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priority 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Otherwise the node is somewhere on the left-top edge...
                    else {

                      // Priority 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Otherwise the node is neither on the left-bottom edge nor on the left-top edge.
                  else {

                    // If the node is somewhere on the left-rear edge.
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 3 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priority 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // No cells around.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is somewhere on the left-front edge.
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priority 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Otherwise the node is somewhere on the left face...
                    else {

                      // Priority 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Parent_cell is the owner.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }
                }

                // From here, we have explored all nodes and all edges of the left parent face.

                // If the node is on the right face of the parent cell.
                else if (i == (child_coord.x + pattern - 1) && (!mask_node_if_cell_right[l])) {

                  // If the node is on the bottom face of the parent cell.
                  // So the node is on the right-bottom edge.
                  if (j == child_coord.y && (!mask_node_if_cell_bottom[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So the node is on the right, bottom, and rear (same position as the parent cell node).
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 1 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priority 2 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 2);
                      }

                      // Priority 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Priority 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So the node is on the right, bottom, and front (same position as the parent cell node).
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priority 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Otherwise the node is somewhere on the right-bottom edge...
                    else {

                      // Priority 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priority 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // If the node is on the top face of the parent cell.
                  // So node on the top right edge.
                  else if (j == (child_coord.y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So node on the right, top, rear (same position as the parent cell's node).
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Priority 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Priority 8 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 2);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So node on the right, top, front (same position as the parent cell's node).
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Otherwise the node is somewhere on the top right edge...
                    else {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }

                  // Otherwise the node is neither on the bottom right edge nor on the top right edge.
                  else {
                    // If the node is somewhere on the rear right edge.
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is somewhere on the front right edge.
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Otherwise the node is somewhere on the right face...
                    else {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }
                }

                // From here, we have explored all nodes of the parent cell and all edges
                // of the parent right face (and left).
                // So only four edges and four faces remain to be explored.

                // Otherwise the node is neither on the left face nor on the right face.
                else {

                  // If the node is on the bottom face of the parent cell.
                  if (j == child_coord.y && (!mask_node_if_cell_bottom[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So node on the bottom rear edge.
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 1 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priority 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // No cells around.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So node on the bottom front edge.
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priority 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Otherwise the node is somewhere on the bottom face...
                    else {

                      // Priority 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Parent_cell is the owner.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // If the node is on the top face of the parent cell.
                  else if (j == (child_coord.y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // If the node is on the rear face of the parent cell.
                    // So node on the top rear edge.
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priority 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Otherwise, parent_cell is the owner of the node.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is on the front face of the parent cell.
                    // So node on the top front edge.
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Otherwise the node is somewhere on the top face...
                    else {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }

                  // Only two faces remain, the rear face and the front face...
                  else {

                    // If the node is somewhere on the rear face...
                    if (k == child_coord.z && (!mask_node_if_cell_rear[l])) {

                      // Priority 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Parent_cell is the owner.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // If the node is somewhere on the front face...
                    else if (k == (child_coord.z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Otherwise, the node is inside the parent cell.
                    else {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }
                }
              }

              // If there is a node creation and/or an owner change.
              if (child_node_owner != -1) {
                node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

                // When there is an owner change without node creation,
                // we must set aside the uniqueIds of these nodes to be able to
                // iterate over them later.
                if (!is_new_node) {
                  node_uid_change_owner_only.add(child_nodes_uids[l]);
                  // debug() << "Child node (change owner) -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- node : " << l
                  //         << " -- uid_node : " << child_nodes_uids[l]
                  //         << " -- owner : " << child_node_owner;
                }
                else {
                  // debug() << "Child node (create node)  -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- node : " << l
                  //         << " -- uid_node : " << child_nodes_uids[l]
                  //         << " -- owner : " << child_node_owner;
                }
              }
            }
          }
        }
      }
    }
  }
  else {
    ARCANE_FATAL("Bad dimension");
  }

  // Nodes
  {
    debug() << "Nb new nodes in patch : " << total_nb_nodes;
    {
      const Integer nb_node_owner_change = node_uid_change_owner_only.size();

      // This array will contain the localIds of the new nodes but also the localIds
      // of the nodes that are just changing owners.
      UniqueArray<Int32> nodes_lid(total_nb_nodes + nb_node_owner_change);

      // We create the nodes. We put the localIds of the new nodes at the beginning of the array.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid.subView(0, total_nb_nodes));

      // We look for the localIds of the nodes that change owners and put them at the end of the array.
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(nodes_lid.subView(total_nb_nodes, nb_node_owner_change), node_uid_change_owner_only, true);

      UniqueArray<Int64> uid_child_nodes(total_nb_nodes + nb_node_owner_change);
      UniqueArray<Int32> lid_child_nodes(total_nb_nodes + nb_node_owner_change);
      Integer index = 0;

      // We assign the correct owners to the nodes.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;
        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
        // TODO: Fix this in the directly concerned part.
        else {
          node.mutableItemBase().removeFlags(ItemFlags::II_Shared);
        }
        // Note, node.level() == -1 here.
        uid_child_nodes[index++] = m_num_mng->parentNodeUniqueIdOfNode(node.uniqueId(), max_level + 1, false);
      }

      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(lid_child_nodes, uid_child_nodes, false);
      NodeInfoListView nodes(m_mesh->nodeFamily());

      index = 0;
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        const Int32 child_lid = lid_child_nodes[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Node parent = nodes[child_lid];
        Node child = *inode;

        m_mesh->modifier()->addParentNodeToNode(child, parent);
        m_mesh->modifier()->addChildNodeToNode(parent, child);
      }
    }
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "Nb new faces in patch : " << total_nb_faces;
    {
      const Integer nb_face_owner_change = face_uid_change_owner_only.size();

      // This array will contain the localIds of the new faces but also the localIds
      // of the faces that are just changing owners.
      UniqueArray<Int32> faces_lid(total_nb_faces + nb_face_owner_change);

      // We create the faces. We put the localIds of the new faces at the beginning of the array.
      m_mesh->modifier()->addFaces(total_nb_faces, faces_infos, faces_lid.subView(0, total_nb_faces));

      // We look for the localIds of the faces that change owners and put them at the end of the array.
      m_mesh->faceFamily()->itemsUniqueIdToLocalId(faces_lid.subView(total_nb_faces, nb_face_owner_change), face_uid_change_owner_only, true);

      UniqueArray<Int64> uid_parent_faces(total_nb_faces + nb_face_owner_change);
      UniqueArray<Int32> lid_parent_faces(total_nb_faces + nb_face_owner_change);
      Integer index = 0;

      // We assign the correct owners to the faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;
        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
        // TODO: Fix this in the directly concerned part.
        else {
          face.mutableItemBase().removeFlags(ItemFlags::II_Shared);
        }
        // Note, face.level() == -1 here.
        uid_parent_faces[index++] = m_num_mng->parentFaceUniqueIdOfFace(face.uniqueId(), max_level + 1, false);
        // debug() << "Parent of : " << face.uniqueId() << " is : " << uid_child_faces[index - 1];
      }

      m_mesh->faceFamily()->itemsUniqueIdToLocalId(lid_parent_faces, uid_parent_faces, false);
      FaceInfoListView faces(m_mesh->faceFamily());

      index = 0;
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        const Int32 child_lid = lid_parent_faces[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Face parent = faces[child_lid];
        Face child = *iface;

        m_mesh->modifier()->addParentFaceToFace(child, parent);
        m_mesh->modifier()->addChildFaceToFace(parent, child);
      }
    }
    m_mesh->faceFamily()->notifyItemsOwnerChanged();
  }

  // Cells
  {
    debug() << "Nb new cells in patch : " << total_nb_cells;

    UniqueArray<Int32> cells_lid(total_nb_cells);
    m_mesh->modifier()->addCells(total_nb_cells, cells_infos, cells_lid);

    // Iteration over the new cells.
    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i) {
      Cell child = cells[cells_lid[i]];
      Cell parent = cells[parent_cells[i]];

      child.mutableItemBase().setOwner(parent.owner(), my_rank);

      child.mutableItemBase().addFlags(ItemFlags::II_JustAdded);

      if (parent.owner() == my_rank) {
        child.mutableItemBase().addFlags(ItemFlags::II_Own);
      }

      if (parent.itemBase().flags() & ItemFlags::II_Shared) {
        child.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }

      m_mesh->modifier()->addParentCellToCell(child, parent);
      m_mesh->modifier()->addChildCellToCell(parent, child);
    }

    // Iteration over the parent cells.
    for (Cell cell : cell_to_refine_internals) {
      cell.mutableItemBase().removeFlags(ItemFlags::II_Refine);
      cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined | ItemFlags::II_Inactive);
    }
    m_mesh->cellFamily()->notifyItemsOwnerChanged();
  }

  m_mesh->modifier()->endUpdate();

  // We position the nodes in space.
  for (Cell parent_cell : cell_to_refine_internals) {
    m_num_mng->setChildNodeCoordinates(parent_cell);
    // We add the "II_Shared" flag to the nodes and faces of shared cells.
    if (parent_cell.mutableItemBase().flags() & ItemFlags::II_Shared) {
      for (Int32 i = 0; i < parent_cell.nbHChildren(); ++i) {
        Cell child_cell = parent_cell.hChild(i);
        for (Node node : child_cell.nodes()) {
          if (node.mutableItemBase().flags() & ItemFlags::II_Own) {
            node.mutableItemBase().addFlags(ItemFlags::II_Shared);
          }
        }

        for (Face face : child_cell.faces()) {
          if (face.mutableItemBase().flags() & ItemFlags::II_Own) {
            face.mutableItemBase().addFlags(ItemFlags::II_Shared);
          }
        }
      }
    }
  }

  // Recalculate synchronization information.
  m_mesh->computeSynchronizeInfos();

  //  ENUMERATE_(Cell, icell, m_mesh->allCells()){
  //    debug() << "\t" << *icell;
  //    for(Node node : icell->nodes()){
  //      debug() << "\t\t" << node;
  //    }
  //    for(Face face : icell->faces()){
  //      debug() << "\t\t\t" << face;
  //    }
  //  }
  //  info() << "Summary:";
  //  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
  //    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
  //    for (Integer i = 0; i < icell->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
  //    }
  //  }
  //  info() << "Node summary:";
  //  ENUMERATE_ (Node, inode, m_mesh->allNodes()) {
  //    debug() << "\tNode uniqueId : " << inode->uniqueId() << " -- level : " << inode->level() << " -- nbChildren : " << inode->nbHChildren();
  //    for (Integer i = 0; i < inode->nbHChildren(); ++i) {
  //      debug() << "\t\tNode Child uniqueId : " << inode->hChild(i).uniqueId() << " -- level : " << inode->hChild(i).level() << " -- nbChildren : " << inode->hChild(i).nbHChildren();
  //    }
  //  }
  //
  //  info() << "Résumé :";
  //  ENUMERATE_ (Face, iface, m_mesh->allFaces()) {
  //    debug() << "\tFace uniqueId : " << iface->uniqueId() << " -- level : " << iface->level() << " -- nbChildren : " << iface->nbHChildren();
  //    for (Integer i = 0; i < iface->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << iface->hChild(i).uniqueId() << " -- level : " << iface->hChild(i).level() << " -- nbChildren : " << iface->hChild(i).nbHChildren();
  //    }
  //  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
createSubLevel()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  UniqueArray<Int64> cell_uid_to_create;

  // TODO: Replace around_parent_cells_uid_to_owner with parent_to_child_cells?
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, bool> around_parent_cells_uid_is_in_subdomain;
  std::unordered_map<Int64, UniqueArray<Cell>> parent_to_child_cells;

  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;

  // We are going to create level -1.
  // Note that at the end of the method, we will replace this level
  // at 0.
  m_num_mng->prepareLevel(-1);

  // We create one or more layers of ghost meshes
  // to prevent a parent mesh from not having the same
  // number of child meshes.
  // ----------
  // CartesianMeshCoarsening2::_doDoubleGhostLayers()
  IMeshModifier* mesh_modifier = m_mesh->modifier();
  IGhostLayerMng* gm = m_mesh->ghostLayerMng();
  // We must use version 3 at least to support
  // multiple layers of ghost meshes
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);
  Int32 nb_ghost_layer = gm->nbGhostLayer();
  // TODO AH: This line would allow for fewer ghost meshes and
  // prevent their deletion if unnecessary. But the behavior
  // would be different from the historical AMR.
  //gm->setNbGhostLayer(nb_ghost_layer + (nb_ghost_layer % m_num_mng->pattern()));
  // Historical AMR behavior.
  gm->setNbGhostLayer(nb_ghost_layer * 2);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();
  // Restore the initial number of ghost layers
  gm->setNbGhostLayer(nb_ghost_layer);
  // CartesianMeshCoarsening2::_doDoubleGhostLayers()
  // ----------

  // We retrieve the unique IDs of the parents to be created.
  ENUMERATE_ (Cell, icell, m_mesh->allLevelCells(0)) {
    Cell cell = *icell;

    Int64 parent_uid = m_num_mng->parentCellUniqueIdOfCell(cell);

    // We avoid duplicates.
    if (!cell_uid_to_create.contains(parent_uid)) {
      cell_uid_to_create.add(parent_uid);
      // We take the opportunity to save the owners of the future meshes
      // which will be the same owners as the child meshes.
      around_parent_cells_uid_to_owner[parent_uid] = cell.owner();
      around_parent_cells_uid_is_in_subdomain[parent_uid] = true;
    }
    else {
      // This can happen if the partitioning is not suitable.
      if (around_parent_cells_uid_to_owner[parent_uid] != cell.owner()) {
        ARCANE_FATAL("Pb owner -- Two+ children, two+ different owners, same parent\n"
                     "The ground patch size in x, y (and z if 3D) must be a multiple of four (need partitionner update to support multiple of two)\n"
                     "CellUID : {0} -- CellOwner : {1} -- OtherChildOwner : {2}",
                     cell.uniqueId(), cell.owner(), around_parent_cells_uid_to_owner[parent_uid]);
      }
    }

    // We must save the children of the parents to create the connectivities
    // at the end.
    parent_to_child_cells[parent_uid].add(cell);
  }

  //  info() << cell_uid_to_create;
  //  for (const auto& [key, value] : parent_to_child_cells) {
  //    info() << "Parent : " << key << " -- Children : " << value;
  //  }

  UniqueArray<Int64> cells_infos;
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  // Two arrays allowing retrieval of the unique IDs of nodes and faces
  // for each parent mesh upon every call to getNodeUids()/getFaceUids().
  UniqueArray<Int64> parent_nodes_uids(m_num_mng->nbNodeByCell());
  UniqueArray<Int64> parent_faces_uids(m_num_mng->nbFaceByCell());

  // Part exchanging information about meshes around the patch
  // (to replace ghost meshes).
  {
    // Array that will contain the uids of the meshes whose information we need.
    UniqueArray<Int64> uid_of_cells_needed;
    {
      UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Int64 parent_cell : cell_uid_to_create) {
        m_num_mng->cellUniqueIdsAroundCell(parent_cell, -1, cell_uids_around);
        for (Int64 cell_uid : cell_uids_around) {
          // If -1 then there are no meshes at this position.
          if (cell_uid == -1)
            continue;

          // IF we have the mesh, we do not need to request information.
          if (around_parent_cells_uid_to_owner.contains(cell_uid))
            continue;

          // TODO: Meh
          if (!uid_of_cells_needed.contains(cell_uid)) {
            uid_of_cells_needed.add(cell_uid);

            // If we need the information, it means we don't possess it :-)
            // We take the opportunity to record this information to distinguish between
            // ghost meshes for which we possess the items (faces/nodes) and those for which
            // we possess nothing.
            around_parent_cells_uid_is_in_subdomain[cell_uid] = false;
          }
        }
      }
    }

    // We share the necessary cell uids from everyone.
    UniqueArray<Int64> uid_of_cells_needed_all_procs;
    pm->allGatherVariable(uid_of_cells_needed, uid_of_cells_needed_all_procs);

    UniqueArray<Int32> owner_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());

    {
      // We record the owner of the meshes that we possess.
      for (Integer i = 0; i < uid_of_cells_needed_all_procs.size(); ++i) {
        if (around_parent_cells_uid_to_owner.contains(uid_of_cells_needed_all_procs[i])) {
          owner_of_cells_needed_all_procs[i] = around_parent_cells_uid_to_owner[uid_of_cells_needed_all_procs[i]];
        }
        else {
          // ReduceMax will eliminate this -1.
          owner_of_cells_needed_all_procs[i] = -1;
        }
      }
    }

    // We retrieve the owners of all necessary meshes.
    pm->reduce(Parallel::eReduceType::ReduceMax, owner_of_cells_needed_all_procs);

    // We only process the owners of the necessary meshes for us.
    {
      Integer size_uid_of_cells_needed = uid_of_cells_needed.size();
      Integer my_pos_in_all_procs_arrays = 0;
      UniqueArray<Integer> size_uid_of_cells_needed_per_proc(nb_rank);
      ArrayView<Integer> av(1, &size_uid_of_cells_needed);
      pm->allGather(av, size_uid_of_cells_needed_per_proc);

      // We skip the meshes from all procs before us.
      for (Integer i = 0; i < my_rank; ++i) {
        my_pos_in_all_procs_arrays += size_uid_of_cells_needed_per_proc[i];
      }

      // We record the necessary owners.
      ArrayView<Int32> owner_of_cells_needed = owner_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
      for (Integer i = 0; i < size_uid_of_cells_needed; ++i) {
        around_parent_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];

        // In refinement, there can be multiple levels of differences between patches.
        // In coarsening, this is impossible since level 0 has no "holes."
        if (owner_of_cells_needed[i] == -1) {
          ARCANE_FATAL("In coarsening, this is normally impossible");
        }
      }
    }
  }

  if (m_mesh->dimension() == 2) {

    // Masks allowing us to know if we should create a face/node (true)
    // or if we should look at the adjacent mesh first (false).
    // Reminder that Arcane's face traversal is in the NumPad order {2, 6, 8, 4}.
    constexpr bool mask_face_if_cell_left[] = { true, true, true, false };
    constexpr bool mask_face_if_cell_bottom[] = { false, true, true, true };

    // Reminder that Arcane's node traversal is in the NumPad order {1, 3, 9, 7}.
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true };

    FixedArray<Int64, 9> cells_uid_around;
    FixedArray<Int32, 9> owner_cells_around_parent_cell_1d;
    FixedArray<bool, 9> is_not_in_subdomain_cells_around_parent_cell_1d;

    // For refinement, we would traverse the existing parent meshes.
    // Here, the parent meshes do not exist yet, so we traverse the uids.
    for (Int64 parent_cell_uid : cell_uid_to_create) {

      m_num_mng->cellUniqueIdsAroundCell(parent_cell_uid, -1, cells_uid_around.view());

      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);
      // Be careful with the "not" in the variable name.
      ConstArray2View is_not_in_subdomain_cells_around_parent_cell(is_not_in_subdomain_cells_around_parent_cell_1d.data(), 3, 3);

      for (Integer i = 0; i < 9; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // If uid_cell != -1 then there might be a mesh (but we don't know if it is actually present).
        // If around_parent_cells_uid_to_owner[uid_cell] != -1 then there is indeed a mesh.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = !around_parent_cells_uid_is_in_subdomain[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = true;
        }
      }

      // These two lambdas are different.
      // When a parent_cell does not exist, there is -1 in the appropriate array,
      // so the first lambda will necessarily return true while the second returns false.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(y, x) != -1 && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1)));
      };

      total_nb_cells++;
      // debug() << "Parent"
      //         << " -- x : " << m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, -1)
      //         << " -- y : " << m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, -1)
      //         << " -- level : " << -1
      //         << " -- uid : " << parent_cell_uid;

      // We retrieve the unique IDs of the nodes and faces to be created.
      m_num_mng->cellNodeUniqueIds(parent_cell_uid, -1, parent_nodes_uids);
      m_num_mng->cellFaceUniqueIds(parent_cell_uid, -1, parent_faces_uids);

      constexpr Integer type_cell = IT_Quad4;
      constexpr Integer type_face = IT_Line2;

      // Cell Part.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Face Part.
      for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
        // We check if we should process the face.
        // If mask_face_if_cell_left[l] == false, we must check if the mesh to the left is ours or not
        // or if the mesh to the left is in our subdomain or not.
        // If this mesh is not ours and/or is not in our subdomain,
        // we must create the face as a ghost face.
        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1))) {
          Integer parent_face_owner = -1;
          faces_infos.add(type_face);
          faces_infos.add(parent_faces_uids[l]);

          // The face nodes are always nodes l and l+1
          // because we use the same traversal for both cases.
          for (Integer nc = l; nc < l + 2; nc++) {
            faces_infos.add(parent_nodes_uids[nc % m_num_mng->nbNodeByCell()]);
          }
          total_nb_faces++;

          if ((!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 0);
          }
          else if ((!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(0, 1);
          }
          else {
            parent_face_owner = owner_cells_around_parent_cell(1, 1);
          }
          face_uid_to_owner[parent_faces_uids[l]] = parent_face_owner;
          // debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- face : " << l
          //         << " -- uid_face : " << parent_faces_uids[l]
          //         << " -- owner : " << parent_face_owner;
        }
      }

      // Node Part.
      // This part is quite similar to the face part, apart from the fact that there can be
      // more possible owners.
      for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1))) {
          Integer parent_node_owner = -1;
          nodes_infos.add(parent_nodes_uids[l]);
          total_nb_nodes++;

          if ((!mask_node_if_cell_left[l])) {
            if ((!mask_node_if_cell_bottom[l])) {
              if (is_cell_around_parent_cell_exist_and_different_owner(0, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 0);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 1);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(1, 0);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
            else {
              if (is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(1, 0);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
          }
          else {
            if ((!mask_node_if_cell_bottom[l])) {
              if (is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 1);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 2);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
            else {
              parent_node_owner = owner_cells_around_parent_cell(1, 1);
            }
          }

          node_uid_to_owner[parent_nodes_uids[l]] = parent_node_owner;
          // debug() << "Parent node (create node)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- node : " << l
          //         << " -- uid_node : " << parent_nodes_uids[l]
          //         << " -- owner : " << parent_node_owner;
        }
      }
    }
  }
  else if (m_mesh->dimension() == 3) {

    // Masks allowing us to know if we should create a face/node (true)
    // or if we should look at the adjacent mesh first (false).
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false, false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true, false, false, true, true };
    constexpr bool mask_node_if_cell_rear[] = { false, false, false, false, true, true, true, true };

    constexpr bool mask_face_if_cell_left[] = { true, false, true, true, true, true };
    constexpr bool mask_face_if_cell_bottom[] = { true, true, false, true, true, true };
    constexpr bool mask_face_if_cell_rear[] = { false, true, true, true, true, true };

    // Small difference compared to 2D. For 2D, the position of the face nodes
    // in the "parent_nodes_uids" array is always the same (l and l+1, see 2D).
    // For 3D, this is not the case, so we have arrays to have a correspondence
    // between the nodes of each face and the position of the nodes in the "parent_nodes_uids" array.
    // (Example: for face 1 (same enumeration order as Arcane), we must take the
    // "nodes_in_face_1" array and thus the nodes "parent_nodes_uids[0]", "parent_nodes_uids[3]",
    // "parent_nodes_uids[7]" and "parent_nodes_uids[4]").
    constexpr Integer nodes_in_face_0[] = { 0, 1, 2, 3 };
    constexpr Integer nodes_in_face_1[] = { 0, 3, 7, 4 };
    constexpr Integer nodes_in_face_2[] = { 0, 1, 5, 4 };
    constexpr Integer nodes_in_face_3[] = { 4, 5, 6, 7 };
    constexpr Integer nodes_in_face_4[] = { 1, 2, 6, 5 };
    constexpr Integer nodes_in_face_5[] = { 3, 2, 6, 7 };

    constexpr Integer nb_nodes_in_face = 4;
    FixedArray<Int64, 27> cells_uid_around;
    FixedArray<Int32, 27> owner_cells_around_parent_cell_1d;
    FixedArray<bool, 27> is_not_in_subdomain_cells_around_parent_cell_1d;

    // For refinement, we would traverse the existing parent meshes.
    // Here, the parent meshes do not exist yet, so we traverse the uids.
    for (Int64 parent_cell_uid : cell_uid_to_create) {

      m_num_mng->cellUniqueIdsAroundCell(parent_cell_uid, -1, cells_uid_around.view());

      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);
      // Be careful with the "not" in the variable name.
      ConstArray3View is_not_in_subdomain_cells_around_parent_cell(is_not_in_subdomain_cells_around_parent_cell_1d.data(), 3, 3, 3);

      for (Integer i = 0; i < 27; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // If uid_cell != -1 then there might be a mesh (but we don't know if it is actually present).
        // If around_parent_cells_uid_to_owner[uid_cell] != -1 then there is indeed a mesh.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = !around_parent_cells_uid_is_in_subdomain[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = true;
        }
      }

      // These two lambdas are different.
      // When a parent_cell does not exist, there is -1 in the appropriate array,
      // so the first lambda will necessarily return true while the second returns false.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != -1 && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1)));
      };

      total_nb_cells++;
      // debug() << "Parent"
      //         << " -- x : " << m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, -1)
      //         << " -- y : " << m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, -1)
      //         << " -- z : " << m_num_mng->cellUniqueIdToCoordZ(parent_cell_uid, -1)
      //         << " -- level : " << -1
      //         << " -- uid : " << parent_cell_uid;

      // We retrieve the unique IDs of the nodes and faces to be created.
      m_num_mng->cellNodeUniqueIds(parent_cell_uid, -1, parent_nodes_uids);
      m_num_mng->cellFaceUniqueIds(parent_cell_uid, -1, parent_faces_uids);

      constexpr Integer type_cell = IT_Hexaedron8;
      constexpr Integer type_face = IT_Quad4;

      // Cell Part.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Face Part.
      for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
        // We check if we should process the face.
        // If mask_face_if_cell_left[l] == false, we must check if the cell on the left belongs to us or not
        // or if the cell on the left is in our subdomain or not.
        // If this cell is not ours and/or is not in our subdomain,
        // we must create the face as a ghost face.
        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1) || is_not_in_subdomain_cells_around_parent_cell(1, 0, 1)) &&
        (mask_face_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1, 1))) {
          Integer parent_face_owner = -1;
          faces_infos.add(type_face);
          faces_infos.add(parent_faces_uids[l]);

          // We retrieve the position of the face nodes in the "ua_node_uid" array.
          ConstArrayView<Integer> nodes_in_face_l;
          switch (l) {
          case 0:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_0, nb_nodes_in_face);
            break;
          case 1:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_1, nb_nodes_in_face);
            break;
          case 2:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_2, nb_nodes_in_face);
            break;
          case 3:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_3, nb_nodes_in_face);
            break;
          case 4:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_4, nb_nodes_in_face);
            break;
          case 5:
            nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_5, nb_nodes_in_face);
            break;
          default:
            ARCANE_FATAL("Bizarre...");
          }
          for (Integer nc : nodes_in_face_l) {
            faces_infos.add(parent_nodes_uids[nc]);
          }
          total_nb_faces++;

          if ((!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 1, 0);
          }
          else if ((!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 0, 1);
          }
          else if ((!mask_face_if_cell_rear[l]) && is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(0, 1, 1);
          }
          else {
            parent_face_owner = owner_cells_around_parent_cell(1, 1, 1);
          }
          face_uid_to_owner[parent_faces_uids[l]] = parent_face_owner;
          // debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- face : " << l
          //         << " -- uid_face : " << parent_faces_uids[l]
          //         << " -- owner : " << parent_face_owner;
        }
      }

      // Node Part.
      // This part is quite similar to the face part, except that there can be
      // more possible owners.
      for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1) || is_not_in_subdomain_cells_around_parent_cell(1, 0, 1)) &&
        (mask_node_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1, 1))) {
          Integer parent_node_owner = -1;
          nodes_infos.add(parent_nodes_uids[l]);
          total_nb_nodes++;

          if ((!mask_node_if_cell_left[l])) {
            if ((!mask_node_if_cell_bottom[l])) {
              if ((!mask_node_if_cell_rear[l])) {

                if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
            else {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
          }
          else {
            if ((!mask_node_if_cell_bottom[l])) {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
            else {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
              }
            }
          }

          node_uid_to_owner[parent_nodes_uids[l]] = parent_node_owner;
          // debug() << "Parent node (create node)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- node : " << l
          //         << " -- uid_node : " << parent_nodes_uids[l]
          //         << " -- owner : " << parent_node_owner;
        }
      }
    }
  }
  else {
    ARCANE_FATAL("Bad dimension");
  }

  // Nodes
  {
    debug() << "Nb new nodes in patch : " << total_nb_nodes;
    {
      // This array will contain the localIds of the new nodes.
      UniqueArray<Int32> nodes_lid(total_nb_nodes);

      // We create the nodes. We put the localIds of the new nodes at the beginning of the array.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid);

      UniqueArray<Int64> uid_child_nodes(total_nb_nodes);
      UniqueArray<Int32> lid_child_nodes(total_nb_nodes);
      Integer index = 0;

      // We assign the correct owners to the nodes.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;

        ARCANE_ASSERT((node_uid_to_owner.contains(node.uniqueId())), ("No owner found for node"));
        ARCANE_ASSERT((node_uid_to_owner[node.uniqueId()] < nb_rank && node_uid_to_owner[node.uniqueId()] >= 0), ("Bad owner found for node"));

        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }

        uid_child_nodes[index++] = m_num_mng->childNodeUniqueIdOfNode(node.uniqueId(), -1);
      }
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(lid_child_nodes, uid_child_nodes, false);
      NodeInfoListView nodes(m_mesh->nodeFamily());

      index = 0;
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        const Int32 child_lid = lid_child_nodes[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Node child = nodes[child_lid];
        Node parent = *inode;

        m_mesh->modifier()->addParentNodeToNode(child, parent);
        m_mesh->modifier()->addChildNodeToNode(parent, child);
      }
    }

    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "Nb new faces in patch : " << total_nb_faces;
    {
      Integer nb_child = (m_mesh->dimension() == 2 ? 2 : 4);
      UniqueArray<Int32> faces_lid(total_nb_faces);

      m_mesh->modifier()->addFaces(total_nb_faces, faces_infos, faces_lid);

      UniqueArray<Int64> uid_child_faces(total_nb_faces * m_num_mng->nbFaceByCell());
      UniqueArray<Int32> lid_child_faces(total_nb_faces * m_num_mng->nbFaceByCell());
      Integer index = 0;

      // We assign the correct owners to the faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;

        ARCANE_ASSERT((face_uid_to_owner.contains(face.uniqueId())), ("No owner found for face"));
        ARCANE_ASSERT((face_uid_to_owner[face.uniqueId()] < nb_rank && face_uid_to_owner[face.uniqueId()] >= 0), ("Bad owner found for face"));

        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }

        for (Integer i = 0; i < nb_child; ++i) {
          uid_child_faces[index++] = m_num_mng->childFaceUniqueIdOfFace(face.uniqueId(), -1, i);
        }
      }

      m_mesh->faceFamily()->itemsUniqueIdToLocalId(lid_child_faces, uid_child_faces, false);
      FaceInfoListView faces(m_mesh->faceFamily());

      index = 0;
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        for (Integer i = 0; i < nb_child; ++i) {
          const Int32 child_lid = lid_child_faces[index++];
          if (child_lid == NULL_ITEM_ID) {
            continue;
          }

          Face child = faces[child_lid];
          Face parent = *iface;

          m_mesh->modifier()->addParentFaceToFace(child, parent);
          m_mesh->modifier()->addChildFaceToFace(parent, child);
        }
      }
    }

    m_mesh->faceFamily()->notifyItemsOwnerChanged();
  }

  // Cells
  UniqueArray<Int32> cells_lid(total_nb_cells);
  {
    debug() << "Nb new cells in patch : " << total_nb_cells;

    m_mesh->modifier()->addCells(total_nb_cells, cells_infos, cells_lid);

    // Iterating over the new cells.
    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i) {
      Cell parent = cells[cells_lid[i]];

      parent.mutableItemBase().setOwner(around_parent_cells_uid_to_owner[parent.uniqueId()], my_rank);

      parent.mutableItemBase().addFlags(ItemFlags::II_JustAdded);
      parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
      parent.mutableItemBase().addFlags(ItemFlags::II_Inactive);

      if (around_parent_cells_uid_to_owner[parent.uniqueId()] == my_rank) {
        parent.mutableItemBase().addFlags(ItemFlags::II_Own);
      }
      if (parent_to_child_cells[parent.uniqueId()][0].itemBase().flags() & ItemFlags::II_Shared) {
        parent.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }
      for (Cell child : parent_to_child_cells[parent.uniqueId()]) {
        m_mesh->modifier()->addParentCellToCell(child, parent);
        m_mesh->modifier()->addChildCellToCell(parent, child);
      }
    }
    m_mesh->cellFamily()->notifyItemsOwnerChanged();
  }

  m_mesh->modifier()->endUpdate();
  m_num_mng->updateFirstLevel();

  // We position the nodes in space.
  CellInfoListView cells(m_mesh->cellFamily());
  for (Integer i = 0; i < total_nb_cells; ++i) {
    Cell parent_cell = cells[cells_lid[i]];
    m_num_mng->setParentNodeCoordinates(parent_cell);

    // We add the "II_Shared" flag to the nodes and faces of shared cells.
    if (parent_cell.mutableItemBase().flags() & ItemFlags::II_Shared) {
      for (Node node : parent_cell.nodes()) {
        if (node.mutableItemBase().flags() & ItemFlags::II_Own) {
          node.mutableItemBase().addFlags(ItemFlags::II_Shared);
        }
      }
      for (Face face : parent_cell.faces()) {
        if (face.mutableItemBase().flags() & ItemFlags::II_Own) {
          face.mutableItemBase().addFlags(ItemFlags::II_Shared);
        }
      }
    }
  }

  // Recalculate synchronization information
  // This is not necessary for AMR because this information will be recalculated
  // during refinement, but since we do not know if we will perform refinement
  // afterward, it is better to calculate this information in all cases.
  m_mesh->computeSynchronizeInfos();

  //  ENUMERATE_(Cell, icell, m_mesh->allCells()){
  //    debug() << "\t" << *icell;
  //    for(Node node : icell->nodes()){
  //      debug() << "\t\t" << node;
  //    }
  //    for(Face face : icell->faces()){
  //      debug() << "\t\t\t" << face;
  //    }
  //  }
  //  info() << "Summary:";
  //  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
  //    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
  //    for (Integer i = 0; i < icell->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
  //    }
  //  }
  //  info() << "Node summary:";
  //  ENUMERATE_ (Node, inode, m_mesh->allNodes()) {
  //    debug() << "\tNode uniqueId : " << inode->uniqueId() << " -- level : " << inode->level() << " -- nbChildren : " << inode->nbHChildren();
  //    for (Integer i = 0; i < inode->nbHChildren(); ++i) {
  //      debug() << "\t\tNode Child uniqueId : " << inode->hChild(i).uniqueId() << " -- level : " << inode->hChild(i).level() << " -- nbChildren : " << inode->hChild(i).nbHChildren();
  //    }
  //  }
  //
  //  info() << "Face summary:";
  //  ENUMERATE_ (Face, iface, m_mesh->allFaces()) {
  //    debug() << "\tFace uniqueId : " << iface->uniqueId() << " -- level : " << iface->level() << " -- nbChildren : " << iface->nbHChildren();
  //    for (Integer i = 0; i < iface->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << iface->hChild(i).uniqueId() << " -- level : " << iface->hChild(i).level() << " -- nbChildren : " << iface->hChild(i).nbHChildren();
  //    }
  //  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
coarsen(bool update_parent_flag)
{
  // We start by listing the cells to coarsen.
  UniqueArray<Cell> cells_to_coarsen_internal;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Coarsen) {
      if (cell.level() == 0) {
        ARCANE_FATAL("Cannot coarse level-0 cell");
      }

      Cell parent = cell.hParent();

      if (update_parent_flag) {
        parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
        parent.mutableItemBase().removeFlags(ItemFlags::II_Inactive);
        parent.mutableItemBase().removeFlags(ItemFlags::II_CoarsenInactive);
      }

      // For a level n-1 cell, if one of its child cells must be coarsened,
      // then all its child cells must be coarsened.
      for (Integer i = 0; i < parent.nbHChildren(); ++i) {
        Cell child = parent.hChild(i);
        if (!(child.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
          ARCANE_FATAL("Parent cannot have children with coarse flag and children without coarse flag -- Parent uid: {0} -- Child uid: {1}", parent.uniqueId(), child.uniqueId());
        }
      }
      if (parent.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
        ARCANE_FATAL("Cannot coarse parent and child in same time");
      }
      if (cell.nbHChildren() != 0) {
        ARCANE_FATAL("For now, cannot coarse cell with children");
      }
      cells_to_coarsen_internal.add(cell);
    }
  }

  // Maps replacing ghost cells.
  std::unordered_map<Int64, Integer> around_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_cells_uid_to_flags;

  {
    // We only need these two flags for surrounding cells.
    // (II_Coarsen to know if surrounding cells are also to be de-refined)
    // (II_Inactive to know if surrounding cells are already refined (to check that there is no more than one level of difference))
    Int32 useful_flags = ItemFlags::II_Coarsen + ItemFlags::II_Inactive;
    _shareInfosOfCellsAroundPatch(cells_to_coarsen_internal, around_cells_uid_to_owner, around_cells_uid_to_flags, useful_flags);
  }

  // Before deleting the cells, we must change the owners of the faces/nodes between the
  // cells to be deleted and the remaining cells.
  if (m_mesh->dimension() == 2) {
    FixedArray<Int64, 9> uid_cells_around_cell_1d;
    FixedArray<Int32, 9> owner_cells_around_cell_1d;
    FixedArray<Int32, 9> flags_cells_around_cell_1d;

    for (Cell cell_to_coarsen : cells_to_coarsen_internal) {
      const Int64 cell_to_coarsen_uid = cell_to_coarsen.uniqueId();
      m_num_mng->cellUniqueIdsAroundCell(cell_to_coarsen, uid_cells_around_cell_1d.view());

      {
        Integer nb_cells_to_coarsen_or_empty_around = 0;

        for (Integer i = 0; i < 9; ++i) {
          Int64 uid_cell = uid_cells_around_cell_1d[i];
          // If uid_cell != -1 then there might be a cell (but we don't know if it is actually present).
          // If around_cells_uid_to_owner[uid_cell] != -1 then there is indeed a cell.
          if (uid_cell != -1 && around_cells_uid_to_owner[uid_cell] != -1) {
            owner_cells_around_cell_1d[i] = around_cells_uid_to_owner[uid_cell];
            flags_cells_around_cell_1d[i] = around_cells_uid_to_flags[uid_cell];

            if (flags_cells_around_cell_1d[i] & ItemFlags::II_Coarsen) {
              nb_cells_to_coarsen_or_empty_around++;
            }
          }
          else {
            uid_cells_around_cell_1d[i] = -1;
            owner_cells_around_cell_1d[i] = -1;
            flags_cells_around_cell_1d[i] = 0;

            nb_cells_to_coarsen_or_empty_around++;
          }
        }

        // If all cells around us are either non-existent or
        // to be deleted, there is no need to look for new owners for our items.
        // Our cell is to be de-refined, so nb_cells_to_coarsen_or_empty_around >= 1.
        if (nb_cells_to_coarsen_or_empty_around == 9) {
          continue;
        }
      }

      // The owner of our cell.
      Int32 cell_to_coarsen_owner = owner_cells_around_cell_1d[4];

      {
        // We provide the face position in the surrounding cells array
        // according to the face index assigned by Arcane.
        // (See comment tagged "arcane_order_to_around_2d").
        // cell_to_coarsen.face(0) = uid_cells_around_cell_1d[1]
        // cell_to_coarsen.face(1) = uid_cells_around_cell_1d[5]
        // ...
        constexpr Integer arcane_order_to_pos_around[] = { 1, 5, 7, 3 };

        Integer count = -1;

        for (Face face : cell_to_coarsen.faces()) {
          count++;
          Int64 other_cell_uid = uid_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_uid == -1) {
            // We are at the boundary of the mesh or there is no cell of the same level next to it,
            // no need to change the face owner (it will be deleted).
            continue;
          }
          Int32 other_cell_flag = flags_cells_around_cell_1d[arcane_order_to_pos_around[count]];

          if (other_cell_flag & ItemFlags::II_Coarsen) {
            // The adjacent cell will also be deleted, no need to change the face owner.
            continue;
          }
          if (other_cell_flag & ItemFlags::II_Inactive) {
            // The adjacent cell has children. There will therefore be at least two levels of
            // refinement difference.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
          }
          Int32 other_cell_owner = owner_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_owner != cell_to_coarsen_owner) {
            // The adjacent cell exists and belongs to someone else. We give it the face.
            face.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
          }
        }
      }

      {
        Integer count = -1;

        // Here, it's more complicated.
        // Each element of the level 0 array designates a node.
        // Like for faces, the order is described in the comment tagged "arcane_order_to_around_2d".
        //
        // Furthermore, there is the priority aspect.
        // A node is present in four cells. Three cells will potentially be "surviving".
        // We must determine who this node will belong to among these three cells.
        // To do this, we use the priorities described in the comment
        // tagged "priority_owner_2d".
        //
        // Example: Node #0 is present on four surrounding cells with priorities: P0, P1, P3, P4.
        //           Cell P4 (ours) will be deleted.
        //           In array #0, we put the three priorities (from lowest to highest).
        //           Then, we iterate over these three cells (always from lowest to highest).
        //           If cell i is "surviving", it takes ownership.
        //           Finally, the cell with the highest priority will have the node.
        //           If no cell takes ownership, then the node will be deleted.
        //
        constexpr Integer priority_and_pos_of_cells_around_node[4][3] = { { 3, 1, 0 }, { 5, 2, 1 }, { 8, 7, 5 }, { 7, 6, 3 } };

        for (Node node : cell_to_coarsen.nodes()) {
          count++;
          Integer final_owner = -1;
          for (Integer other_cell = 0; other_cell < 3; ++other_cell) {
            Int64 other_cell_uid = uid_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_uid == -1) {
              // We are at the boundary of the mesh or there is no cell of the same level next to it,
              // the node will not take an owner.
              continue;
            }
            Int32 other_cell_flag = flags_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];

            if (other_cell_flag & ItemFlags::II_Coarsen) {
              // The adjacent cell will also be deleted, it cannot take
              // ownership of our node.
              continue;
            }
            if (other_cell_flag & ItemFlags::II_Inactive) {
              // The adjacent cell has children. There will therefore be at least two levels of
              // refinement difference.
              ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
            }
            Int32 other_cell_owner = owner_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_owner != cell_to_coarsen_owner) {
              // The adjacent cell exists and belongs to someone else. We give it the node.
              final_owner = other_cell_owner;
            }
          }
          if (final_owner != -1) {
            node.mutableItemBase().setOwner(final_owner, cell_to_coarsen_owner);
          }
        }
      }
    }
  }
  else if (m_mesh->dimension() == 3) {
    FixedArray<Int64, 27> uid_cells_around_cell_1d;
    FixedArray<Int32, 27> owner_cells_around_cell_1d;
    FixedArray<Int32, 27> flags_cells_around_cell_1d;

    for (Cell cell_to_coarsen : cells_to_coarsen_internal) {
      const Int64 cell_to_coarsen_uid = cell_to_coarsen.uniqueId();
      m_num_mng->cellUniqueIdsAroundCell(cell_to_coarsen, uid_cells_around_cell_1d.view());

      {
        Integer nb_cells_to_coarsen_or_empty_around = 0;

        for (Integer i = 0; i < 27; ++i) {
          Int64 uid_cell = uid_cells_around_cell_1d[i];
          // If uid_cell != -1 then there might be a cell (but we don't know if it is actually present).
          // If around_cells_uid_to_owner[uid_cell] != -1 then there is indeed a cell.
          if (uid_cell != -1 && around_cells_uid_to_owner[uid_cell] != -1) {
            owner_cells_around_cell_1d[i] = around_cells_uid_to_owner[uid_cell];
            flags_cells_around_cell_1d[i] = around_cells_uid_to_flags[uid_cell];

            if (flags_cells_around_cell_1d[i] & ItemFlags::II_Coarsen) {
              nb_cells_to_coarsen_or_empty_around++;
            }
          }
          else {
            uid_cells_around_cell_1d[i] = -1;
            owner_cells_around_cell_1d[i] = -1;
            flags_cells_around_cell_1d[i] = 0;

            nb_cells_to_coarsen_or_empty_around++;
          }
        }

        // If all cells around us are either non-existent or
        // to be deleted, there is no need to look for new owners for our items.
        // Our cell is to be de-refined, so nb_cells_to_coarsen_or_empty_around >= 1.
        if (nb_cells_to_coarsen_or_empty_around == 27) {
          continue;
        }
      }

      // The owner of our cell.
      Int32 cell_to_coarsen_owner = owner_cells_around_cell_1d[13];

      {
        // We provide the face position in the surrounding cells array
        // according to the face index assigned by Arcane.
        // (See comment tagged "arcane_order_to_around_3d").
        // cell_to_coarsen.face(0) = uid_cells_around_cell_1d[4]
        // cell_to_coarsen.face(1) = uid_cells_around_cell_1d[12]
        // ...
        constexpr Integer arcane_order_to_pos_around[] = { 4, 12, 10, 22, 14, 16 };

        Integer count = -1;

        for (Face face : cell_to_coarsen.faces()) {
          count++;
          Int64 other_cell_uid = uid_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_uid == -1) {
            // We are at the boundary of the mesh or there is no cell of the same level next to it,
            // no need to change the face owner (it will be deleted).
            continue;
          }
          Int32 other_cell_flag = flags_cells_around_cell_1d[arcane_order_to_pos_around[count]];

          if (other_cell_flag & ItemFlags::II_Coarsen) {
            // The adjacent cell will also be deleted, no need to change the face owner.
            continue;
          }
          if (other_cell_flag & ItemFlags::II_Inactive) {
            // The adjacent cell has children. There will therefore be at least two levels of
            // refinement difference.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
          }
          Int32 other_cell_owner = owner_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_owner != cell_to_coarsen_owner) {
            // The adjacent cell exists and belongs to someone else. We give it the face.
            face.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
          }
        }
      }

      {
        Integer count = -1;

        // Each element of the level 0 array designates a node.
        // Like for faces, the order is described in the comment tagged "arcane_order_to_around_3d".
        //
        // Furthermore, there is the priority aspect.
        // A node is present in eight cells. Seven cells will potentially be "surviving".
        // We must determine who this node will belong to among these seven cells.
        // To do this, we use the priorities described in the comment
        // tagged "priority_owner_3d".
        //
        // Example: Node #0 is present on eight surrounding cells with priorities: P12, P10, P9, ...
        //           Cell P13 (ours) will be deleted.
        //           In array #0, we put the seven priorities (from lowest to highest).
        //           Then, we iterate over these seven cells (always from lowest to highest).
        //           If cell i is "surviving", it takes ownership.
        //           Finally, the cell with the highest priority will have the node.
        //           If no cell takes ownership, then the node will be deleted.
        //
        constexpr Integer priority_and_pos_of_cells_around_node[8][7] = {
          {12, 10, 9, 4, 3, 1, 0},
          {14, 11, 10, 5, 4, 2, 1},
          {17, 16, 14, 8, 7, 5, 4},
          {16, 15, 12, 7, 6, 4, 3},
          {22, 21, 19, 18, 12, 10, 9},
          {23, 22, 20, 19, 14, 11, 10},
          {26, 25, 23, 22, 17, 16, 14},
          {25, 24, 22, 21, 16, 15, 12}
        };

        for (Node node : cell_to_coarsen.nodes()) {
          count++;
          Integer final_owner = -1;
          for (Integer other_cell = 0; other_cell < 7; ++other_cell) {
            Int64 other_cell_uid = uid_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_uid == -1) {
              // We are at the boundary of the mesh or there is no cell of the same level next to it,
              // the node will not take an owner.
              continue;
            }
            Int32 other_cell_flag = flags_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];

            if (other_cell_flag & ItemFlags::II_Coarsen) {
              // The adjacent cell will also be deleted, it cannot take
              // ownership of our node.
              continue;
            }
            if (other_cell_flag & ItemFlags::II_Inactive) {
              // The adjacent cell has children. There will therefore be at least two levels of
              // refinement difference.
              ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
            }
            Int32 other_cell_owner = owner_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_owner != cell_to_coarsen_owner) {
              // The adjacent cell exists and belongs to someone else. We give it the node.
              node.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
            }
          }
          if (final_owner != -1) {
            node.mutableItemBase().setOwner(final_owner, cell_to_coarsen_owner);
          }
        }
      }
    }
  }

  else {
    ARCANE_FATAL("Bad dimension");
  }

  UniqueArray<Int32> local_ids;
  for (Cell cell : cells_to_coarsen_internal) {
    local_ids.add(cell.localId());
  }
  m_mesh->modifier()->removeCells(local_ids);
  m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  m_mesh->faceFamily()->notifyItemsOwnerChanged();
  m_mesh->modifier()->endUpdate();
  m_mesh->cellFamily()->computeSynchronizeInfos();
  m_mesh->nodeFamily()->computeSynchronizeInfos();
  m_mesh->faceFamily()->computeSynchronizeInfos();
  m_mesh->modifier()->setDynamic(true);

  UniqueArray<Int64> ghost_cell_to_refine;
  UniqueArray<Int64> ghost_cell_to_coarsen;

  if (!update_parent_flag) {
    // If materials are active, a material recalculation must be forced because the
    // cell groups have been modified, and thus the list of constituents as well.
    Materials::IMeshMaterialMng* mm = Materials::IMeshMaterialMng::getReference(m_mesh, false);
    if (mm)
      mm->forceRecompute();
  }

  m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine, ghost_cell_to_coarsen, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Method allowing retrieval of owners and flags of
 * cells around patch_cells.
 *
 * \param patch_cells The cells around which we need the surrounding cell information.
 * \param around_cells_uid_to_owner Owners of patch_cells and cells around patch_cells.
 * \param around_cells_uid_to_flags Flags of patch_cells and cells around patch_cells.
 * \param useful_flags The flags that need to be retrieved.
 */
void CartesianMeshAMRPatchMng::
_shareInfosOfCellsAroundPatch(ConstArrayView<Cell> patch_cells, std::unordered_map<Int64, Integer>& around_cells_uid_to_owner, std::unordered_map<Int64, Int32>& around_cells_uid_to_flags, Int32 useful_flags) const
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  // Part exchanging information about cells around the patch
  // (to replace ghost cells).

  // We fill the array with our info, for the other processes.
  ENUMERATE_ (Cell, icell, m_mesh->ownCells()) {
    Cell cell = *icell;
    around_cells_uid_to_owner[cell.uniqueId()] = my_rank;
    around_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & useful_flags) + ItemFlags::II_UserMark1);
  }

  ENUMERATE_ (Cell, icell, m_mesh->allCells().ghost()) {
    Cell cell = *icell;
    around_cells_uid_to_owner[cell.uniqueId()] = cell.owner();
    around_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & useful_flags) + ItemFlags::II_UserMark1);
  }

  // Array that will contain the uids of the cells whose info we need.
  UniqueArray<Int64> uid_of_cells_needed;
  {
    UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
    for (Cell cell : patch_cells) {
      m_num_mng->cellUniqueIdsAroundCell(cell, cell_uids_around);
      for (Int64 cell_uid : cell_uids_around) {
        // If -1 then there are no cells at this position.
        if (cell_uid == -1)
          continue;

        // IF we have the cell, we don't need to request info.
        if (around_cells_uid_to_owner.contains(cell_uid))
          continue;

        uid_of_cells_needed.add(cell_uid);
      }
    }
  }

  UniqueArray<Int64> uid_of_cells_needed_all_procs;
  pm->allGatherVariable(uid_of_cells_needed, uid_of_cells_needed_all_procs);

  UniqueArray<Int32> flags_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());
  UniqueArray<Int32> owner_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());

  {
    UniqueArray<Int32> local_ids(uid_of_cells_needed_all_procs.size());
    m_mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids, uid_of_cells_needed_all_procs, false);
    Integer compt = 0;
    ENUMERATE_ (Cell, icell, m_mesh->cellFamily()->view(local_ids)) {
      // isOwn is important since there might be ghost cells.
      if (!icell->null() && icell->isOwn()) {
        owner_of_cells_needed_all_procs[compt] = my_rank;
        flags_of_cells_needed_all_procs[compt] = (icell->itemBase().flags() & useful_flags);
      }
      else {
        owner_of_cells_needed_all_procs[compt] = -1;
        flags_of_cells_needed_all_procs[compt] = 0;
      }
      compt++;
    }
  }

  pm->reduce(Parallel::eReduceType::ReduceMax, owner_of_cells_needed_all_procs);
  pm->reduce(Parallel::eReduceType::ReduceMax, flags_of_cells_needed_all_procs);

  // From this point, if the parent_cells are at level 0, the array
  // "owner_of_cells_needed_all_procs" should no longer contain "-1".
  // If the parent_cells are at level 1 or higher, there may be "-1"
  // because the surrounding cells are not necessarily all refined.
  // (example: we are doing level 2, so we look at level 1 parent cells around.
  // It is possible that the adjacent cell has never been refined, so it does not have
  // level 1 cells. Since the cell does not exist, no process can set an owner,
  // so the owner array will contain "-1").

  // We retrieve the info of the surrounding cells that interest us.
  {
    Integer size_uid_of_cells_needed = uid_of_cells_needed.size();
    Integer my_pos_in_all_procs_arrays = 0;
    UniqueArray<Integer> size_uid_of_cells_needed_per_proc(nb_rank);
    ArrayView<Integer> av(1, &size_uid_of_cells_needed);
    pm->allGather(av, size_uid_of_cells_needed_per_proc);

    for (Integer i = 0; i < my_rank; ++i) {
      my_pos_in_all_procs_arrays += size_uid_of_cells_needed_per_proc[i];
    }

    ArrayView<Int32> owner_of_cells_needed = owner_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
    ArrayView<Int32> flags_of_cells_needed = flags_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
    for (Integer i = 0; i < size_uid_of_cells_needed; ++i) {
      around_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];
      around_cells_uid_to_flags[uid_of_cells_needed[i]] = flags_of_cells_needed[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
