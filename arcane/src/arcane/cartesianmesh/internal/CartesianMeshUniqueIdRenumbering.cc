// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUniqueIdRenumbering.cc                         (C) 2000-2024 */
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
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/MeshUtils.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshUniqueIdRenumbering::NewUniqueIdList
{
 public:

  NewUniqueIdList(IMesh* mesh)
  : cells(VariableBuildInfo(mesh, "ArcaneRenumberCellsNewUid"))
  , nodes(VariableBuildInfo(mesh, "ArcaneRenumberNodesNewUid"))
  , faces(VariableBuildInfo(mesh, "ArcaneRenumberFacesNewUid"))
  {
    cells.fill(NULL_ITEM_UNIQUE_ID);
    nodes.fill(NULL_ITEM_UNIQUE_ID);
    faces.fill(NULL_ITEM_UNIQUE_ID);
  }

 public:

  void markCellNoRenumber(Cell c)
  {
    _setCell(c);
    for (Node n : c.nodes())
      _setNode(n);
    for (Face f : c.faces())
      _setFace(f);
  }
  Int64 maxUniqueId() const
  {
    return math::max(max_cell_uid, max_node_uid, max_face_uid);
  }

 public:

  VariableCellInt64 cells;
  VariableNodeInt64 nodes;
  VariableFaceInt64 faces;

 private:

  Int64 max_cell_uid = NULL_ITEM_UNIQUE_ID;
  Int64 max_node_uid = NULL_ITEM_UNIQUE_ID;
  Int64 max_face_uid = NULL_ITEM_UNIQUE_ID;

 private:

  void _setNode(Node node)
  {
    Int64 uid = node.uniqueId();
    nodes[node] = uid;
    if (uid > max_node_uid)
      max_node_uid = uid;
  }
  void _setFace(Face face)
  {
    Int64 uid = face.uniqueId();
    faces[face] = uid;
    if (uid > max_face_uid)
      max_face_uid = uid;
  }
  void _setCell(Cell cell)
  {
    Int64 uid = cell.uniqueId();
    cells[cell] = uid;
    if (uid > max_cell_uid)
      max_cell_uid = uid;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshUniqueIdRenumbering::
CartesianMeshUniqueIdRenumbering(ICartesianMesh* cmesh, ICartesianMeshGenerationInfo* gen_info,
                                 CartesianPatch parent_patch, Int32 patch_method)
: TraceAccessor(cmesh->traceMng())
, m_cartesian_mesh(cmesh)
, m_generation_info(gen_info)
, m_parent_patch(parent_patch)
, m_patch_method(patch_method)
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
  IParallelMng* pm = mesh->parallelMng();
  const Int32 dimension = mesh->dimension();
  Int64 cartesian_global_nb_cell = m_generation_info->globalNbCell();

  const bool use_v2 = (m_patch_method == 3);
  info() << "Apply UniqueId renumbering to mesh '" << mesh->name() << "'"
         << " global_nb_cell=" << cartesian_global_nb_cell
         << " global_nb_cell_by_dim=" << m_generation_info->globalNbCells()
         << " mesh_dimension=" << dimension
         << " patch_method=" << m_patch_method
         << " use_v2=" << use_v2;

  NewUniqueIdList new_uids(mesh);

  // Indique si on utilise le patch par défaut.
  bool use_default_patch = false;
  CartesianPatch patch0 = m_parent_patch;
  if (patch0.isNull()) {
    patch0 = m_cartesian_mesh->patch(0);
    if (m_patch_method != 4)
      use_default_patch = true;
  }

  // TODO: Afficher le numéro du patch.

  // Marque les entités issues de 'm_parent_patch' comme n'étant pas renumérotées.
  // Si 'm_parent_patch' n'est pas spécifié, on prend les mailles du patch initial.
  // Marque aussi les mailles parentes du patch comme étant non renumérotées.
  // NOTE: Cela ne fonctionne bien que si 'm_parent_patch' est le patch initial
  // ou un patch juste en dessous. Dans l'utilisation actuelle c'est toujours le cas
  // car on appelle cette méthode avec soit le patch de base, soit le patch issu
  // du raffinement du patch initial.
  Int32 patch_level = 0;
  ENUMERATE_ (Cell, icell, patch0.cells()) {
    Cell c{ *icell };
    patch_level = c.level();
    new_uids.markCellNoRenumber(c);
    Cell parent = c.hParent();
    if (!parent.null())
      new_uids.markCellNoRenumber(parent);
  }

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

  // Calcule le nombre de mailles du patch servant de référence.
  // Suppose qu'on raffine d'un facteur 2 à chaque fois
  info() << "PatchLevel=" << patch_level << " use_default_path=" << use_default_patch;
  Int32 multiplier = 1;
  for (Int32 z = 0; z < patch_level; ++z)
    multiplier *= 2;
  nb_cell_x *= multiplier;
  nb_cell_y *= multiplier;
  nb_cell_z *= multiplier;

  CartesianGridDimension grid(nb_cell_x, nb_cell_y, nb_cell_z);

  Int64 max_item_uid = pm->reduce(Parallel::ReduceMax, new_uids.maxUniqueId());
  info() << "MaxItem uniqueId=" << max_item_uid;
  Int64 base_adder = 1 + max_item_uid;
  if (use_default_patch)
    base_adder = 0;

  // On suppose que la patch servant de référence a une numérotation cartésienne d'origine
  // ce qui veut dire qu'on peut déterminer les coordonnées topologiques de la maille
  // grâce à son uniqueId()
  if (dimension == 2) {
    CartesianGridDimension::CellUniqueIdComputer2D cell_uid_computer(0, nb_cell_x);
    ENUMERATE_ (Cell, icell, patch0.cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      Int32 level = cell.level();
      auto [coord_i, coord_j, coord_k] = cell_uid_computer.compute(uid);
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << uid << " I=" << coord_i
               << " J=" << coord_j << " nb_cell_x=" << nb_cell_x;
      if (use_default_patch)
        // Indique qu'on est de niveau 1 pour avoir la même numérotation qu'avec la version 3.9
        level = 1;
      _applyChildrenCell2D(cell, new_uids, coord_i, coord_j, nb_cell_x, nb_cell_y, level, base_adder);
    }
  }
  else if (dimension == 3) {
    CartesianGridDimension::CellUniqueIdComputer3D cell_uid_computer(grid.getCellComputer3D(0));
    ENUMERATE_ (Cell, icell, patch0.cells()) {
      Cell cell{ *icell };
      Int64 uid = cell.uniqueId();
      auto [coord_i, coord_j, coord_k] = cell_uid_computer.compute(uid);
      Int32 level = cell.level();
      if (m_is_verbose)
        info() << "Renumbering: PARENT: cell_uid=" << uid << " level=" << level
               << " I=" << coord_i << " J=" << coord_j << " K=" << coord_k
               << " nb_cell_x=" << nb_cell_x << " nb_cell_y=" << nb_cell_y << " nb_cell_z=" << nb_cell_z;
      if (use_v2)
        _applyChildrenCell3DV2(cell, new_uids, coord_i, coord_j, coord_k,
                               nb_cell_x, nb_cell_y, nb_cell_z, level,
                               base_adder, base_adder, base_adder);
      else
        _applyChildrenCell3D(cell, new_uids, coord_i, coord_j, coord_k,
                             nb_cell_x, nb_cell_y, nb_cell_z, level,
                             base_adder, base_adder, base_adder);
    }
  }

  // TODO: faire une classe pour cela.
  //info() << "Change CellFamily";
  //mesh->cellFamily()->notifyItemsUniqueIdChanged();

  _applyFamilyRenumbering(mesh->cellFamily(), new_uids.cells);
  _applyFamilyRenumbering(mesh->nodeFamily(), new_uids.nodes);
  _applyFamilyRenumbering(mesh->faceFamily(), new_uids.faces);
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
_applyChildrenCell2D(Cell cell, NewUniqueIdList& new_uids,
                     Int64 coord_i, Int64 coord_j,
                     Int64 nb_cell_x, Int64 nb_cell_y, Int32 level, Int64 base_adder)
{
  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 4,
  // il faudrait prendre le max des nbHChildren()

  if (coord_i >= nb_cell_x)
    ARCANE_FATAL("Bad coordinate X={0} max={1}", coord_i, nb_cell_x);
  if (coord_j >= nb_cell_y)
    ARCANE_FATAL("Bad coordinate Y={0} max={1}", coord_j, nb_cell_y);

  // Suppose qu'on a un pattern 2x2
  coord_i *= 2;
  coord_j *= 2;
  nb_cell_x *= 2;
  nb_cell_y *= 2;
  const Int64 nb_node_x = nb_cell_x + 1;
  const Int64 nb_node_y = nb_cell_y + 1;
  const Int64 cell_adder = base_adder + (nb_cell_x * nb_cell_y * level);
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 node_adder = base_adder + (nb_node_x * nb_node_y * level);
  const Int64 face_adder = base_adder + (node_adder * 2);

  // Renumérote les noeuds de la maille courante.
  // Suppose qu'on a 4 noeuds
  // ATTENTION a priori on ne peut pas conserver facilement l'ordre
  // des uniqueId() entre l'ancienne et la nouvelle numérotation.
  // Cela invalide l'orientation des faces qu'il faudra refaire.
  {
    if (cell.nbNode() != 4)
      ARCANE_FATAL("Invalid number of nodes N={0}, expected=4", cell.nbNode());
    std::array<Int64, 4> node_uids;
    node_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_node_x);
    node_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_node_x);
    node_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_node_x);
    node_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_node_x);
    for (Integer z = 0; z < 4; ++z) {
      Node node = cell.node(z);
      if (new_uids.nodes[node] < 0) {
        node_uids[z] += node_adder;
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << node_uids[z];
        new_uids.nodes[node] = node_uids[z];
      }
    }
  }
  // Renumérote les faces
  // TODO: Vérifier la validité de cette méthode.
  {
    if (cell.nbFace() != 4)
      ARCANE_FATAL("Invalid number of faces N={0}, expected=4", cell.nbFace());
    std::array<Int64, 4> face_uids;
    face_uids[0] = (coord_i + 0) + ((coord_j + 0) * nb_face_x);
    face_uids[1] = (coord_i + 1) + ((coord_j + 0) * nb_face_x);
    face_uids[2] = (coord_i + 1) + ((coord_j + 1) * nb_face_x);
    face_uids[3] = (coord_i + 0) + ((coord_j + 1) * nb_face_x);
    for (Integer z = 0; z < 4; ++z) {
      Face face = cell.face(z);
      if (new_uids.faces[face] < 0) {
        face_uids[z] += face_adder;
        if (m_is_verbose)
          info() << "APPLY_FACE_CHILD: uid=" << face.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << face_uids[z];
        new_uids.faces[face] = face_uids[z];
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

    _applyChildrenCell2D(sub_cell, new_uids, my_coord_i, my_coord_j,
                         nb_cell_x, nb_cell_y, level + 1, base_adder);
    if (new_uids.cells[sub_cell] < 0)
      new_uids.cells[sub_cell] = new_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
_applyChildrenCell3D(Cell cell, NewUniqueIdList& new_uids,
                     Int64 coord_i, Int64 coord_j, Int64 coord_k,
                     Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int64 current_level_nb_cell_z,
                     Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder)
{
  if (coord_i >= current_level_nb_cell_x)
    ARCANE_FATAL("Bad coordinate X={0} max={1}", coord_i, current_level_nb_cell_x);
  if (coord_j >= current_level_nb_cell_y)
    ARCANE_FATAL("Bad coordinate Y={0} max={1}", coord_j, current_level_nb_cell_y);
  if (coord_k >= current_level_nb_cell_z)
    ARCANE_FATAL("Bad coordinate Z={0} max={1}", coord_k, current_level_nb_cell_z);

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
    if (new_uids.cells[cell] < 0) {
      new_uids.cells[cell] = new_uid;
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
    std::array<Int64, 8> node_uids;
    node_uids[0] = (coord_i + 0) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[1] = (coord_i + 1) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[2] = (coord_i + 1) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[3] = (coord_i + 0) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 0) * current_level_nb_node_x * current_level_nb_node_y);

    node_uids[4] = (coord_i + 0) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[5] = (coord_i + 1) + ((coord_j + 0) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[6] = (coord_i + 1) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);
    node_uids[7] = (coord_i + 0) + ((coord_j + 1) * current_level_nb_node_x) + ((coord_k + 1) * current_level_nb_node_x * current_level_nb_node_y);

    for (Integer z = 0; z < 8; ++z) {
      Node node = cell.node(z);
      if (new_uids.nodes[node] < 0) {
        node_uids[z] += node_adder;
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << node_uids[z];
        new_uids.nodes[node] = node_uids[z];
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
    std::array<Int64, 6> face_uids;

    //// Version originale :
    // face_uids[0] = (coord_k * current_level_nb_cell_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i);

    // face_uids[3] = ((coord_k+1) * current_level_nb_cell_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i);

    // face_uids[1] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_face_x)
    //             + (coord_i) + total_face_xy;

    // face_uids[4] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y)
    //             + (coord_j * current_level_nb_face_x)
    //             + (coord_i+1) + total_face_xy;

    // face_uids[2] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y)
    //             + (coord_j * current_level_nb_cell_x)
    //             + (coord_i) + total_face_xy_yz;

    // face_uids[5] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y)
    //             + ((coord_j+1) * current_level_nb_cell_x)
    //             + (coord_i) + total_face_xy_yz;
    ////

    const Int64 nb_cell_before_j = coord_j * current_level_nb_cell_x;

    face_uids[0] = (coord_k * current_level_nb_cell_x * current_level_nb_cell_y) + nb_cell_before_j + (coord_i);

    face_uids[3] = face_uids[0] + current_level_nb_cell_x * current_level_nb_cell_y;

    face_uids[1] = (coord_k * current_level_nb_face_x * current_level_nb_cell_y) + (coord_j * current_level_nb_face_x) + (coord_i) + total_face_xy;

    face_uids[4] = face_uids[1] + 1;

    face_uids[2] = (coord_k * current_level_nb_cell_x * current_level_nb_face_y) + nb_cell_before_j + (coord_i) + total_face_xy_yz;

    face_uids[5] = face_uids[2] + current_level_nb_cell_x;

    for (Integer z = 0; z < 6; ++z) {
      Face face = cell.face(z);
      if (new_uids.faces[face] < 0) {
        face_uids[z] += face_adder;
        if (m_is_verbose)
          info() << "APPLY_FACE_CHILD: uid=" << face.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << face_uids[z];
        new_uids.faces[face] = face_uids[z];
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

    _applyChildrenCell3D(sub_cell, new_uids, my_coord_i, my_coord_j, my_coord_k,
                         current_level_nb_cell_x, current_level_nb_cell_y, current_level_nb_cell_z,
                         current_level, cell_adder, node_adder, face_adder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshUniqueIdRenumbering::
_applyChildrenCell3DV2(Cell cell, NewUniqueIdList& new_uids,
                       Int64 coord_i, Int64 coord_j, Int64 coord_k,
                       Int64 current_level_nb_cell_x, Int64 current_level_nb_cell_y, Int64 current_level_nb_cell_z,
                       Int32 current_level, Int64 cell_adder, Int64 node_adder, Int64 face_adder)
{
  if (coord_i >= current_level_nb_cell_x)
    ARCANE_FATAL("Bad coordinate X={0} max={1}", coord_i, current_level_nb_cell_x);
  if (coord_j >= current_level_nb_cell_y)
    ARCANE_FATAL("Bad coordinate Y={0} max={1}", coord_j, current_level_nb_cell_y);
  if (coord_k >= current_level_nb_cell_z)
    ARCANE_FATAL("Bad coordinate Z={0} max={1}", coord_k, current_level_nb_cell_z);

  coord_i *= 2;
  coord_j *= 2;
  coord_k *= 2;

  current_level_nb_cell_x *= 2;
  current_level_nb_cell_y *= 2;
  current_level_nb_cell_z *= 2;

  CartesianGridDimension grid(current_level_nb_cell_x, current_level_nb_cell_y, current_level_nb_cell_z);
  CartesianGridDimension::CellUniqueIdComputer3D cell_uid_computer(grid.getCellComputer3D(cell_adder));
  CartesianGridDimension::FaceUniqueIdComputer3D face_uid_computer(grid.getFaceComputer3D(face_adder));
  CartesianGridDimension::NodeUniqueIdComputer3D node_uid_computer(grid.getNodeComputer3D(node_adder));

  // TODO: pour pouvoir s'adapter à tous les raffinements, au lieu de 8,
  // il faudrait prendre le max des nbHChildren()

  Int64x3 grid_nb_node = grid.nbNode();
  Int64x3 grid_nb_face = grid.nbFace();

  // Renumérote la maille.
  {
    Int64 new_uid = cell_uid_computer.compute(coord_i, coord_j, coord_k);
    if (new_uids.cells[cell] < 0) {
      new_uids.cells[cell] = new_uid;
      if (m_is_verbose)
        info() << "APPLY_CELL_CHILD: uid=" << cell.uniqueId() << " I=" << coord_i << " J=" << coord_j << " K=" << coord_k
               << " current_level=" << current_level << " new_uid=" << new_uid << " CellAdder=" << cell_adder;
    }
  }

  static constexpr Int32 const_cell_nb_node = 8;
  // Renumérote les noeuds de la maille courante.
  {
    std::array<Int64, const_cell_nb_node> node_uids = node_uid_computer.computeForCell(coord_i, coord_j, coord_k);

    for (Integer z = 0; z < const_cell_nb_node; ++z) {
      Node node = cell.node(z);
      if (new_uids.nodes[node] < 0) {
        if (m_is_verbose)
          info() << "APPLY_NODE_CHILD: uid=" << node.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << node_uids[z];
        new_uids.nodes[node] = node_uids[z];
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
  //const Int64 total_face_xy = grid_nb_face.z * current_level_nb_cell_x * current_level_nb_cell_y;
  //const Int64 total_face_xy_yz = total_face_xy + grid_nb_face.x * current_level_nb_cell_y * current_level_nb_cell_z;
  //const Int64 total_face_xy_yz_zx = total_face_xy_yz + grid_nb_face.y * current_level_nb_cell_z * current_level_nb_cell_x;
  {
    std::array<Int64, 6> face_uids = face_uid_computer.computeForCell(coord_i, coord_j, coord_k);

    for (Integer z = 0; z < 6; ++z) {
      Face face = cell.face(z);
      if (new_uids.faces[face] < 0) {
        const bool do_print = false;
        if (do_print) {
          info() << "Parent_cell=" << cell.uniqueId() << " level=" << cell.level()
                 << " face_adder=" << face_adder << " z=" << z
                 << " x=" << coord_i << " y=" << coord_j << " z=" << coord_k
                 << " cx=" << current_level_nb_cell_x << " cy=" << current_level_nb_cell_y << " cz=" << current_level_nb_cell_z;
        }
        if (m_is_verbose || do_print)
          info() << "APPLY_FACE_CHILD: uid=" << face.uniqueId() << " parent_cell=" << cell.uniqueId()
                 << " I=" << z << " new_uid=" << face_uids[z];
        new_uids.faces[face] = face_uids[z];
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
  cell_adder += grid.totalNbCell();
  node_adder += grid_nb_node.x * grid_nb_node.y * grid_nb_node.z;
  face_adder += grid_nb_face.x * grid_nb_face.y * grid_nb_face.z;

  current_level += 1;

  Int32 nb_child = cell.nbHChildren();
  for (Int32 icell = 0; icell < nb_child; ++icell) {
    Cell sub_cell = cell.hChild(icell);
    Int64 my_coord_i = coord_i + icell % 2;
    Int64 my_coord_j = coord_j + (icell % 4) / 2;
    Int64 my_coord_k = coord_k + icell / 4;

    _applyChildrenCell3DV2(sub_cell, new_uids, my_coord_i, my_coord_j, my_coord_k,
                           current_level_nb_cell_x, current_level_nb_cell_y, current_level_nb_cell_z,
                           current_level, cell_adder, node_adder, face_adder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
