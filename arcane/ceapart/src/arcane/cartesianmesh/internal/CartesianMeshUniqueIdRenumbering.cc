// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.cc                         (C) 2000-2021 */
/*                                                                           */
/* Renumérotation des uniqueId() pour les maillages cartésiens.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianMeshUniqueIdRenumbering.h"

#include "arcane/utils/PlatformUtils.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/ICartesianMeshPatch.h"

#include "arcane/VariableTypes.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ICartesianMeshGenerationInfo.h"

#include <array>

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
  if (platform::getEnvironmentVariable("ARCANE_DEBUG_AMR_RENUMBERING") == "1")
    m_is_verbose = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
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

  // Marque les entités issues du maillage cartésien comme étant de niveau 0
  // Elles ne seront pas renumérotées
  ICartesianMeshPatch* patch0 = m_cartesian_mesh->patch(0);
  ENUMERATE_ (Cell, icell, patch0->cells()) {
    Cell c{ *icell };
    cells_new_uid[icell] = c.uniqueId().asInt64();
    for (Node n : c.nodes())
      nodes_new_uid[n] = n.uniqueId();
    for (Face f : c.faces())
      faces_new_uid[f] = f.uniqueId();
  }

  // Pour chaque maille de niveau 0, calcule son indice (i,j) dans le maillage cartésien

  // Pour cela, on suppose que le maillage a été créé avec le 'CartesianMeshGenerator'
  // (ou un générateur qui a la même numérotation) et que le uniqueId() d'une maille est:
  //   Int64 cell_unique_id = i + j * all_nb_cell_x;
  // avec:
  //   all_nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];

  // En 3D :
  //   Int64 cell_unique_id = i + j * all_nb_cell_x + k * (all_nb_cell_x * all_nb_cell_y);
  // avec:
  //   all_nb_cell_x = m_generation_info->globalNbCells()[MD_DirX];
  //   all_nb_cell_y = m_generation_info->globalNbCells()[MD_DirY];

  Int64ConstArrayView global_nb_cells_by_direction = m_generation_info->globalNbCells();
  Int64 nb_cell_x = global_nb_cells_by_direction[MD_DirX];
  if (nb_cell_x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", nb_cell_x);

  Int64 nb_cell_y = global_nb_cells_by_direction[MD_DirY];
  if (nb_cell_y <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", nb_cell_y);

  Int64 nb_cell_z = global_nb_cells_by_direction[MD_DirZ];
  if (nb_cell_z <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", nb_cell_z);

  if(dimension == 2){
    ENUMERATE_ (Cell, icell, patch0->cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      Int64 coord_i = uid % nb_cell_x;
      Int64 coord_j = uid / nb_cell_x;
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << cell.uniqueId() << " I=" << coord_i
               << " J=" << coord_j << " nb_cell_x=" << nb_cell_x;
      _applyChildrenCell2D(cell, nodes_new_uid, faces_new_uid, cells_new_uid, coord_i, coord_j, nb_cell_x, nb_cell_y, 0);
    }
  }

  else if(dimension == 3){
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
      _applyChildrenCell3D(cell, nodes_new_uid, faces_new_uid, cells_new_uid, coord_i, coord_j, coord_k, nb_cell_x, nb_cell_y, nb_cell_z, 1);
    }
  }

  
  // TODO: faire une classe pour cela.
  //info() << "Change CellFamily";
  //mesh->cellFamily()->notifyItemsUniqueIdChanged();

  _applyFamilyRenumbering(mesh->cellFamily(), cells_new_uid);
  _applyFamilyRenumbering(mesh->nodeFamily(), nodes_new_uid);
  _applyFamilyRenumbering(mesh->faceFamily(), faces_new_uid);
  mesh->checkValidMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
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

void CartesianMeshUniqueIdRenumbering::
_applyChildrenCell2D(Cell cell, VariableNodeInt64& nodes_new_uid, VariableFaceInt64& faces_new_uid,
                   VariableCellInt64& cells_new_uid,
                   Int64 coord_i, Int64 coord_j,
                   Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int32 current_level)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 4,
  // il faudrait prendre le max des nbHChildren()

  // Suppose qu'on a un pattern 2x2
  const Int64 parent_level_nb_cell_x = current_level_nb_cell_x/2;
  const Int64 parent_level_nb_cell_y = current_level_nb_cell_y/2;

  const Int64 cell_adder = parent_level_nb_cell_x * parent_level_nb_cell_y * current_level;
  const Int64 nb_node_x = current_level_nb_cell_x + 1;
  const Int64 nb_node_y = current_level_nb_cell_y + 1;
  const Int64 node_adder = (parent_level_nb_cell_x+1) * (parent_level_nb_cell_y+1) * current_level;
  current_level_nb_cell_x *= 2;
  current_level_nb_cell_y *= 2;
  //coord_i *= 2;
  //coord_j *= 2;
  const Int64 nb_face_x = current_level_nb_cell_x + 1;
  const Int64 face_adder = node_adder * 2;

  // Renumérote les noeuds de la maille courante.
  // Suppose qu'on a 4 noeuds
  // ATTENTION a priori on ne peut pas conserver facilement l'ordre
  // des uniqueId() entre l'ancienne et la nouvelle numérotation.
  // Cela invalide l'orientation des faces qu'il faudra refaire.
  {
    if (cell.nbNode() != 4)
      ARCANE_FATAL("Invalid number of nodes N={0}, expected=4", cell.nbNode());
    std::array<Int64, 4> new_uids;

    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_node_x);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_node_x);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_node_x);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_node_x);

    for (Integer z = 0; z < 4; ++z) {
      Node node = cell.node(z);
      //if (nodes_new_uid[node] < 0) {
        new_uids[z] += node_adder;
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << new_uids[z];
        nodes_new_uid[node] = new_uids[z];
      //}
    }
  }
  // Renumérote les faces
  // TODO: Vérifier la validité de cette méthode.
  {
    if (cell.nbFace() != 4)
      ARCANE_FATAL("Invalid number of faces N={0}, expected=4", cell.nbFace());
    std::array<Int64, 4> new_uids;
    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_face_x);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_face_x);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_face_x);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_face_x);
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
  // Renumérote les sous-mailles
  // Suppose qu'on a 4 mailles enfants comme suit par mailles
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i*2 + icell % 2;
    Int64 my_coord_j = coord_j*2 + icell / 2;
    Int64 new_uid = (my_coord_i + my_coord_j * current_level_nb_cell_x) + cell_adder;
    if (m_is_verbose)
      info() << "APPLY_CELL_CHILD: uid=" << sub_cell.uniqueId() << " I=" << my_coord_i << " J=" << my_coord_j
             << " current_level=" << current_level << " new_uid=" << new_uid << " NodeAdder=" << node_adder;

    _applyChildrenCell2D(sub_cell, nodes_new_uid, faces_new_uid, cells_new_uid, my_coord_i, my_coord_j,
                         current_level_nb_cell_x, current_level_nb_cell_y, current_level + 1);
    if (cells_new_uid[sub_cell] < 0)
      cells_new_uid[sub_cell] = new_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
_applyChildrenCell3D(Cell cell, VariableNodeInt64& nodes_new_uid, VariableFaceInt64& faces_new_uid,
                   VariableCellInt64& cells_new_uid,
                   Int64 coord_i, Int64 coord_j, Int64 coord_k,
                   Int64 nb_cell_x, Int64 nb_cell_y, Int64 nb_cell_z, Int32 level)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 8,
  // il faudrait prendre le max des nbHChildren()

  // Suppose qu'on a un pattern 2x2x2
  coord_i *= 2;
  coord_j *= 2;
  coord_k *= 2;
  nb_cell_x *= 2;
  nb_cell_y *= 2;
  nb_cell_z *= 2;
  const Int64 nb_node_x = nb_cell_x + 1;
  const Int64 nb_node_y = nb_cell_y + 1;
  const Int64 nb_node_z = nb_cell_z + 1;
  const Int64 cell_adder = nb_cell_x * nb_cell_y * nb_cell_z * level;
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 nb_face_y = nb_cell_y + 1;
  const Int64 node_adder = nb_node_x * nb_node_y * nb_node_z * level;
  const Int64 face_adder = node_adder * 4;

  // Renumérote les noeuds de la maille courante.
  // Suppose qu'on a 8 noeuds
  // ATTENTION a priori on ne peut pas conserver facilement l'ordre
  // des uniqueId() entre l'ancienne et la nouvelle numérotation.
  // Cela invalide l'orientation des faces qu'il faudra refaire.
  {
    if (cell.nbNode() != 8)
      ARCANE_FATAL("Invalid number of nodes N={0}, expected=8", cell.nbNode());
    std::array<Int64, 8> new_uids;
    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_node_x) + ((coord_k + 0) * nb_node_x * nb_node_y);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_node_x) + ((coord_k + 0) * nb_node_x * nb_node_y);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_node_x) + ((coord_k + 0) * nb_node_x * nb_node_y);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_node_x) + ((coord_k + 0) * nb_node_x * nb_node_y);

    new_uids[4] = (coord_i + 0) + ((coord_j + 0) * nb_node_x) + ((coord_k + 1) * nb_node_x * nb_node_y);
    new_uids[5] = (coord_i + 1) + ((coord_j + 0) * nb_node_x) + ((coord_k + 1) * nb_node_x * nb_node_y);
    new_uids[6] = (coord_i + 1) + ((coord_j + 1) * nb_node_x) + ((coord_k + 1) * nb_node_x * nb_node_y);
    new_uids[7] = (coord_i + 0) + ((coord_j + 1) * nb_node_x) + ((coord_k + 1) * nb_node_x * nb_node_y);

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
  // Renumérote les faces
  // TODO: Vérifier la validité de cette méthode.
  {
    if (cell.nbFace() != 6)
      ARCANE_FATAL("Invalid number of faces N={0}, expected=6", cell.nbFace());
    std::array<Int64, 6> new_uids;
    new_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_face_x) + ((coord_k + 0) * nb_face_x * nb_face_y * 2);
    new_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_face_x) + ((coord_k + 0) * nb_face_x * nb_face_y * 2);
    new_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_face_x) + ((coord_k + 0) * nb_face_x * nb_face_y * 2);
    new_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_face_x) + ((coord_k + 0) * nb_face_x * nb_face_y * 2);

    new_uids[4] = (coord_i + 0) + ((coord_j + 0) * nb_face_x) + ((coord_k + 1) * nb_face_x * nb_face_y * 2);
    new_uids[5] = (coord_i + 1) + ((coord_j + 0) * nb_face_x) + ((coord_k + 1) * nb_face_x * nb_face_y * 2);

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
  // Renumérote les sous-mailles
  // Suppose qu'on a 8 mailles enfants (2x2x2) comme suit par mailles
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i + icell % 2;
    Int64 my_coord_j = coord_j + icell / 2;
    Int64 my_coord_k = coord_k + icell / 4; // x*y
    Int64 new_uid = (my_coord_i + my_coord_j * nb_cell_x + my_coord_k * nb_cell_x * nb_cell_y * 2) + cell_adder;
    if (m_is_verbose)
      info() << "APPLY_CELL_CHILD: uid=" << sub_cell.uniqueId() << " I=" << my_coord_i << " J=" << my_coord_j << " K=" << my_coord_k
             << " level=" << level << " new_uid=" << new_uid << " NodeAdder=" << node_adder;

    _applyChildrenCell3D(sub_cell, nodes_new_uid, faces_new_uid, cells_new_uid, my_coord_i, my_coord_j, my_coord_k,
                         nb_cell_x, nb_cell_y, nb_cell_z, level + 1);
    if (cells_new_uid[sub_cell] < 0)
      cells_new_uid[sub_cell] = new_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

