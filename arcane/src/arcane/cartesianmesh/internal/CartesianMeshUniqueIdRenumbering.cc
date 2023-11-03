// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.cc                         (C) 2000-2023 */
/*                                                                           */
/* Renumérotation des uniqueId() pour les maillages cartésiens.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianMeshUniqueIdRenumbering.h"

#include "arcane/utils/PlatformUtils.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/ICartesianMeshPatch.h"

#include "arcane/core/CartesianGridDimension.h"
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
  const Int32 dimension = mesh->dimension();
  Int64 cartesian_global_nb_cell = m_generation_info->globalNbCell();
  info() << "Apply UniqueId renumbering to mesh '" << mesh->name() << "'"
         << " global_nb_cell=" << cartesian_global_nb_cell
         << " global_nb_cell_by_dim=" << m_generation_info->globalNbCells()
         << " mesh_dimension=" << dimension;

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
  Int64 nb_cell_y = global_nb_cells_by_direction[MD_DirY];
  Int64 nb_cell_z = global_nb_cells_by_direction[MD_DirZ];

  if (nb_cell_x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", nb_cell_x);
  if (dimension >= 2 && nb_cell_y <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", nb_cell_y);
  if (dimension >= 3 && nb_cell_z <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", nb_cell_z);

  if (dimension == 2) {
    CartesianGridDimension::CellUniqueIdComputer2D cell_uid_computer(0, nb_cell_x);
    ENUMERATE_ (Cell, icell, patch0->cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      auto [coord_i, coord_j, coord_k] = cell_uid_computer.compute(uid);
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << uid << " I=" << coord_i
               << " J=" << coord_j << " nb_cell_x=" << nb_cell_x;
      _applyChildrenCell2D(cell, nodes_new_uid, faces_new_uid, cells_new_uid, coord_i, coord_j, nb_cell_x, nb_cell_y, 1);
    }
  }

  else if (dimension == 3) {
    CartesianGridDimension::CellUniqueIdComputer3D cell_uid_computer(0, nb_cell_x, nb_cell_x * nb_cell_y);
    ENUMERATE_ (Cell, icell, patch0->cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      auto [coord_i, coord_j, coord_k] = cell_uid_computer.compute(uid);
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << uid << " I=" << coord_i
               << " J=" << coord_j << " K=" << coord_k
               << " nb_cell_x=" << nb_cell_x << " nb_cell_y=" << nb_cell_y;
      _applyChildrenCell3D(cell, nodes_new_uid, faces_new_uid, cells_new_uid,
                           coord_i, coord_j, coord_k,
                           nb_cell_x, nb_cell_y, nb_cell_z,
                           0, 0, 0, 0);
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
                     Int64 nb_cell_x, Int64 nb_cell_y, Int32 level)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 4,
  // il faudrait prendre le max des nbHChildren()

  // Suppose qu'on a un pattern 2x2
  coord_i *= 2;
  coord_j *= 2;
  nb_cell_x *= 2;
  nb_cell_y *= 2;
  const Int64 nb_node_x = nb_cell_x + 1;
  const Int64 nb_node_y = nb_cell_y + 1;
  const Int64 cell_adder = nb_cell_x * nb_cell_y * level;
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 node_adder = nb_node_x * nb_node_y * level;
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
    Int64 my_coord_i = coord_i + icell % 2;
    Int64 my_coord_j = coord_j + icell / 2;
    Int64 new_uid = (my_coord_i + my_coord_j * nb_cell_x) + cell_adder;
    if (m_is_verbose)
      info() << "APPLY_CELL_CHILD: uid=" << sub_cell.uniqueId() << " I=" << my_coord_i << " J=" << my_coord_j
             << " level=" << level << " new_uid=" << new_uid << " CellAdder=" << cell_adder;

    _applyChildrenCell2D(sub_cell, nodes_new_uid, faces_new_uid, cells_new_uid, my_coord_i, my_coord_j,
                         nb_cell_x, nb_cell_y, level + 1);
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
                     Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int64 current_level_nb_cell_z,
                     Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 8,
  // il faudrait prendre le max des nbHChildren()

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
  // const Int64 parent_level_nb_cell_x = current_level_nb_cell_x / 2;
  // const Int64 parent_level_nb_cell_y = current_level_nb_cell_y / 2;
  // const Int64 parent_level_nb_cell_z = current_level_nb_cell_z / 2;
  // Int64 level_i_nb_cell_x = parent_level_nb_cell_x;
  // Int64 level_i_nb_cell_y = parent_level_nb_cell_y;
  // Int64 level_i_nb_cell_z = parent_level_nb_cell_z;
  // for(Int32 i = current_level-1; i >= 0; i--){
  //   face_adder += (level_i_nb_cell_z + 1) * level_i_nb_cell_x * level_i_nb_cell_y
  //               + (level_i_nb_cell_x + 1) * level_i_nb_cell_y * level_i_nb_cell_z
  //               + (level_i_nb_cell_y + 1) * level_i_nb_cell_z * level_i_nb_cell_x;

  //   cell_adder += level_i_nb_cell_x * level_i_nb_cell_y * level_i_nb_cell_z;
  //   node_adder += (level_i_nb_cell_x + 1) * (level_i_nb_cell_y + 1) * (level_i_nb_cell_z + 1);

  //   level_i_nb_cell_x /= 2;
  //   level_i_nb_cell_y /= 2;
  //   level_i_nb_cell_z /= 2;
  // }

  // Renumérote la maille.
  {
    Int64 new_uid = (coord_i + coord_j * current_level_nb_cell_x + coord_k * current_level_nb_cell_x * current_level_nb_cell_y) + cell_adder;
    if (cells_new_uid[cell] < 0) {
      cells_new_uid[cell] = new_uid;
      if (m_is_verbose)
        info() << "APPLY_CELL_CHILD: uid=" << cell.uniqueId() << " I=" << coord_i << " J=" << coord_j << " K=" << coord_k
               << " current_level=" << current_level << " new_uid=" << new_uid << " CellAdder=" << cell_adder;
    }
  }

  // Renumérote les noeuds de la maille courante.
  // Suppose qu'on a 8 noeuds
  // ATTENTION a priori on ne peut pas conserver facilement l'ordre
  // des uniqueId() entre l'ancienne et la nouvelle numérotation.
  // Cela invalide l'orientation des faces qu'il faudra refaire.
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

  // Renumérote les faces
  // Cet algo n'est pas basé sur l'algo 2D.
  // Les UniqueIDs générés sont contigües.
  // Il est aussi possible de retrouver les UniqueIDs des faces
  // à l'aide de la position de la cellule et la taille du maillage.
  // De plus, l'ordre des UniqueIDs des faces d'une cellule est toujours le
  // même (en notation localId Arcane (cell.face(i)) : 0, 3, 1, 4, 2, 5).
  // Les UniqueIDs générés sont donc les mêmes quelque soit le découpage.
  /*
       x               z
    ┌──►          │ ┌──►
    │             │ │
   y▼12   13   14 │y▼ ┌────┬────┐
      │ 26 │ 27 │ │   │ 24 │ 25 │
      └────┴────┘ │   0    4    8
     15   16   17 │              
      │ 28 │ 29 │ │   │    │    │
      └────┴────┘ │   2    5    9
   z=0            │              x=0
  - - - - - - - - - - - - - - - - - -
   z=1            │              x=1
     18   19   20 │   ┌────┬────┐
      │ 32 │ 33 │ │   │ 30 │ 31 │
      └────┴────┘ │   1    6   10
     21   22   23 │              
      │ 34 │ 35 │ │   │    │    │
      └────┴────┘ │   3    7   11
                  │
  */
  // On a un cube décomposé en huit cellules (2x2x2).
  // Le schéma au-dessus représente les faces des cellules de ce cube avec
  // les uniqueIDs que l'algorithme génèrera (sans face_adder).
  // Pour cet algo, on commence par les faces "xy".
  // On énumère d'abord en x, puis en y, puis en z.
  // Une fois les faces "xy" numérotées, on fait les faces "yz".
  // Toujours le même ordre de numérotation.
  // On termine avec les faces "zx", encore dans le même ordre.
  //
  // Dans l'implémentation ci-dessous, on fait la numérotation
  // maille par maille.
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

  // Renumérote les sous-mailles
  // Suppose qu'on a 8 mailles enfants (2x2x2) comme suit par mailles
  // -------
  // | 2| 3|
  // -------
  // | 0| 1|
  // -------
  cell_adder += current_level_nb_cell_x * current_level_nb_cell_y * current_level_nb_cell_z;
  node_adder += current_level_nb_node_x * current_level_nb_node_y * current_level_nb_node_z;
  face_adder += total_face_xy_yz_zx;

  coord_i *= 2;
  coord_j *= 2;
  coord_k *= 2;

  current_level_nb_cell_x *= 2;
  current_level_nb_cell_y *= 2;
  current_level_nb_cell_z *= 2;

  current_level += 1;

  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i + icell % 2;
    Int64 my_coord_j = coord_j + (icell % 4) / 2;
    Int64 my_coord_k = coord_k + icell / 4;

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
