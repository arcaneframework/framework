// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.cc                                (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CartesianMeshNumberingMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshNumberingMng::
CartesianMeshNumberingMng(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_pattern(2)
, m_max_level(0)
, m_min_level(0)
{
  auto* m_generation_info = ICartesianMeshGenerationInfo::getReference(m_mesh,true);

  Int64ConstArrayView global_nb_cells_by_direction = m_generation_info->globalNbCells();
  m_nb_cell_x = global_nb_cells_by_direction[MD_DirX];
  if (m_nb_cell_x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", m_nb_cell_x);

  m_nb_cell_y = global_nb_cells_by_direction[MD_DirY];
  if (m_nb_cell_y <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", m_nb_cell_y);

  m_nb_cell_z = global_nb_cells_by_direction[MD_DirZ];
  if (m_nb_cell_z <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", m_nb_cell_z);

  m_p_to_l_level.add(0);

  if (m_mesh->dimension() == 2) {
    m_latest_cell_uid = m_nb_cell_x * m_nb_cell_y;
    m_latest_node_uid = (m_nb_cell_x + 1) * (m_nb_cell_y + 1);
    m_latest_face_uid = (m_nb_cell_x * m_nb_cell_y) * 2 + m_nb_cell_x * 2 + m_nb_cell_y;
  }
  else {
    m_latest_cell_uid = m_nb_cell_x * m_nb_cell_y * m_nb_cell_z;
    m_latest_node_uid = (m_nb_cell_x + 1) * (m_nb_cell_y + 1) * (m_nb_cell_z + 1);
    m_latest_face_uid = (m_nb_cell_z + 1) * m_nb_cell_x * m_nb_cell_y + (m_nb_cell_x + 1) * m_nb_cell_y * m_nb_cell_z + (m_nb_cell_y + 1) * m_nb_cell_z * m_nb_cell_x;
  }

  m_first_cell_uid_level.add(0);
  m_first_node_uid_level.add(0);
  m_first_face_uid_level.add(0);
}

void CartesianMeshNumberingMng::
prepareLevel(Int32 level)
{
  if (level <= m_max_level && level >= m_min_level)
    return;
  if (level == m_max_level + 1) {
    m_max_level++;
  }
  else if (level == m_min_level - 1) {
    m_min_level--;
  }
  else {
    ARCANE_FATAL("Level error : {0}", level);
  }
  m_p_to_l_level.add(level);

  m_first_cell_uid_level.add(m_latest_cell_uid);
  m_first_node_uid_level.add(m_latest_node_uid);
  m_first_face_uid_level.add(m_latest_face_uid);

  Int64 nb_cell_x = getGlobalNbCellsX(level);
  Int64 nb_cell_y = getGlobalNbCellsY(level);
  Int64 nb_cell_z = getGlobalNbCellsZ(level);

  if (m_mesh->dimension() == 2) {
    m_latest_cell_uid += nb_cell_x * nb_cell_y;
    m_latest_node_uid += (nb_cell_x + 1) * (nb_cell_y + 1);
    m_latest_face_uid += (nb_cell_x * nb_cell_y) * 2 + nb_cell_x * 2 + nb_cell_y;
  }
  else {
    m_latest_cell_uid += nb_cell_x * nb_cell_y * nb_cell_z;
    m_latest_node_uid += (nb_cell_x + 1) * (nb_cell_y + 1) * (nb_cell_z + 1);
    m_latest_face_uid += (nb_cell_z + 1) * nb_cell_x * nb_cell_y + (nb_cell_x + 1) * nb_cell_y * nb_cell_z + (nb_cell_y + 1) * nb_cell_z * nb_cell_x;
  }
}

void CartesianMeshNumberingMng::
updateFirstLevel()
{
  Int32 nb_levels_to_add = -m_min_level;

  if (nb_levels_to_add == 0) {
    return;
  }

  m_max_level += nb_levels_to_add;
  m_min_level += nb_levels_to_add;

  for (Int32& i : m_p_to_l_level) {
    i += nb_levels_to_add;
  }

  m_nb_cell_x = m_nb_cell_x / (m_pattern * nb_levels_to_add);
  m_nb_cell_y = m_nb_cell_y / (m_pattern * nb_levels_to_add);
  m_nb_cell_z = m_nb_cell_z / (m_pattern * nb_levels_to_add);
}

Int64 CartesianMeshNumberingMng::
getFirstCellUidLevel(Integer level)
{
  auto pos = m_p_to_l_level.span().findFirst(level);
  if (pos.has_value()) {
    return m_first_cell_uid_level[pos.value()];
  }
  else {
    ARCANE_FATAL("Bad level : {0}", level);
  }
}

Int64 CartesianMeshNumberingMng::
getFirstNodeUidLevel(Integer level)
{
  auto pos = m_p_to_l_level.span().findFirst(level);
  if (pos.has_value()) {
    return m_first_node_uid_level[pos.value()];
  }
  else {
    ARCANE_FATAL("Bad level : {0}", level);
  }
}

Int64 CartesianMeshNumberingMng::
getFirstFaceUidLevel(Integer level)
{
  auto pos = m_p_to_l_level.span().findFirst(level);
  if (pos.has_value()) {
    return m_first_face_uid_level[pos.value()];
  }
  else {
    ARCANE_FATAL("Bad level : {0}", level);
  }
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsX(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell_x) * std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsY(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell_y) * std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsZ(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell_z) * std::pow(m_pattern, level));
}

Integer CartesianMeshNumberingMng::
getPattern() const
{
  return m_pattern;
}

// Tant que l'on a un unique "pattern" pour x, y, z, pas besoin de trois méthodes.
Int64 CartesianMeshNumberingMng::
getOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const
{
  if (level_from == level_to) {
    return coord;
  }
  else if (level_from < level_to) {
    return coord * m_pattern * (level_to - level_from);
  }
  else {
    return coord / (m_pattern * (level_from - level_to));
  }
}

Int64 CartesianMeshNumberingMng::
uidToCoordX(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_cell_y = getGlobalNbCellsY(level);
  const Int64 first_cell_uid = getFirstCellUidLevel(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return to2d % nb_cell_x;
}

Int64 CartesianMeshNumberingMng::
uidToCoordX(Cell cell)
{
  return uidToCoordX(cell.uniqueId(), cell.level());
}

Int64 CartesianMeshNumberingMng::
uidToCoordY(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_cell_y = getGlobalNbCellsY(level);
  const Int64 first_cell_uid = getFirstCellUidLevel(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return to2d / nb_cell_x;
}

Int64 CartesianMeshNumberingMng::
uidToCoordY(Cell cell)
{
  return uidToCoordY(cell.uniqueId(), cell.level());
}

Int64 CartesianMeshNumberingMng::
uidToCoordZ(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_cell_y = getGlobalNbCellsY(level);
  const Int64 first_cell_uid = getFirstCellUidLevel(level);

  uid -= first_cell_uid;

  return uid / (nb_cell_x * nb_cell_y);
}

Int64 CartesianMeshNumberingMng::
uidToCoordZ(Cell cell)
{
  return uidToCoordZ(cell.uniqueId(), cell.level());
}


Int64 CartesianMeshNumberingMng::
getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k)
{
  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_cell_y = getGlobalNbCellsY(level);
  const Int64 first_cell_uid = getFirstCellUidLevel(level);

  return (cell_coord_i + cell_coord_j * nb_cell_x + cell_coord_k * nb_cell_x * nb_cell_y) + first_cell_uid;
}

Int64 CartesianMeshNumberingMng::
getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j)
{
  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 first_cell_uid = getFirstCellUidLevel(level);

  return (cell_coord_i + cell_coord_j * nb_cell_x) + first_cell_uid;
}



Integer CartesianMeshNumberingMng::
getNbNode()
{
  return static_cast<Integer>(std::pow(m_pattern, m_mesh->dimension()));
}

void CartesianMeshNumberingMng::
getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k)
{
  if(uid.size() != getNbNode())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = getGlobalNbCellsX(level) + 1;
  const Int64 nb_node_y = getGlobalNbCellsY(level) + 1;
  const Int64 first_node_uid = getFirstNodeUidLevel(level);

  uid[0] = (cell_coord_i + 0) + ((cell_coord_j + 0) * nb_node_x) + ((cell_coord_k + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[1] = (cell_coord_i + 1) + ((cell_coord_j + 0) * nb_node_x) + ((cell_coord_k + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[2] = (cell_coord_i + 1) + ((cell_coord_j + 1) * nb_node_x) + ((cell_coord_k + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[3] = (cell_coord_i + 0) + ((cell_coord_j + 1) * nb_node_x) + ((cell_coord_k + 0) * nb_node_x * nb_node_y) + first_node_uid;

  uid[4] = (cell_coord_i + 0) + ((cell_coord_j + 0) * nb_node_x) + ((cell_coord_k + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[5] = (cell_coord_i + 1) + ((cell_coord_j + 0) * nb_node_x) + ((cell_coord_k + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[6] = (cell_coord_i + 1) + ((cell_coord_j + 1) * nb_node_x) + ((cell_coord_k + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[7] = (cell_coord_i + 0) + ((cell_coord_j + 1) * nb_node_x) + ((cell_coord_k + 1) * nb_node_x * nb_node_y) + first_node_uid;
}

void CartesianMeshNumberingMng::
getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j)
{
  if(uid.size() != getNbNode())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = getGlobalNbCellsX(level) + 1;
  const Int64 first_node_uid = getFirstNodeUidLevel(level);

  uid[0] = (cell_coord_i + 0) + ((cell_coord_j + 0) * nb_node_x) + first_node_uid;
  uid[1] = (cell_coord_i + 1) + ((cell_coord_j + 0) * nb_node_x) + first_node_uid;
  uid[2] = (cell_coord_i + 1) + ((cell_coord_j + 1) * nb_node_x) + first_node_uid;
  uid[3] = (cell_coord_i + 0) + ((cell_coord_j + 1) * nb_node_x) + first_node_uid;
}

void CartesianMeshNumberingMng::
getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid)
{
  const Int64 cell_coord_i = uidToCoordX(cell_uid, level);
  const Int64 cell_coord_j = uidToCoordY(cell_uid, level);

  if (m_mesh->dimension() == 2) {
    getNodeUids(uid, level, cell_coord_i, cell_coord_j);
  }
  else {
    const Int64 cell_coord_k = uidToCoordZ(cell_uid, level);
    getNodeUids(uid, level, cell_coord_i, cell_coord_j, cell_coord_k);
  }
}

Integer CartesianMeshNumberingMng::
getNbFace()
{
  return m_pattern * m_mesh->dimension();
}

void CartesianMeshNumberingMng::
getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k)
{
  if(uid.size() != getNbFace())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_cell_y = getGlobalNbCellsY(level);
  const Int64 nb_cell_z = getGlobalNbCellsZ(level);

  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 nb_face_y = nb_cell_y + 1;
  const Int64 nb_face_z = nb_cell_z + 1;

  const Int64 first_face_uid = getFirstFaceUidLevel(level);

  // Numérote les faces
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

  const Int64 total_face_xy = nb_face_z * nb_cell_x * nb_cell_y;
  const Int64 total_face_xy_yz = total_face_xy + nb_face_x * nb_cell_y * nb_cell_z;

  const Int64 nb_cell_before_j = cell_coord_j * nb_cell_x;



  uid[0] = (cell_coord_k * nb_cell_x * nb_cell_y)
         + nb_cell_before_j
         + (cell_coord_i);

  uid[3] = uid[0] + nb_cell_x * nb_cell_y;

  uid[1] = (cell_coord_k * nb_face_x * nb_cell_y)
         + (cell_coord_j * nb_face_x)
         + (cell_coord_i) + total_face_xy;

  uid[4] = uid[1] + 1;

  uid[2] = (cell_coord_k * nb_cell_x * nb_face_y)
         + nb_cell_before_j
         + (cell_coord_i) + total_face_xy_yz;

  uid[5] = uid[2] + nb_cell_x;


  uid[0] += first_face_uid;
  uid[1] += first_face_uid;
  uid[2] += first_face_uid;
  uid[3] += first_face_uid;
  uid[4] += first_face_uid;
  uid[5] += first_face_uid;
}


void CartesianMeshNumberingMng::
getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j)
{
  if(uid.size() != getNbFace())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_cell_x = getGlobalNbCellsX(level);
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 first_face_uid = getFirstFaceUidLevel(level);

  // Numérote les faces
  //  |-0--|--2-|
  // 4|   6|   8|
  //  |-5--|-7--|
  // 9|  11|  13|
  //  |-10-|-12-|
  //
  // Avec cette numérotation, HAUT < GAUCHE < BAS < DROITE
  // Mis à part les uniqueIds de la première ligne de face, tous
  // les uniqueIds sont contigües.

  // HAUT
  // - "(current_level_nb_face_x + current_level_nb_cell_x)" :
  //   le nombre de faces GAUCHE BAS DROITE au dessus.
  // - "cell_coord_j * (current_level_nb_face_x + current_level_nb_cell_x)" :
  //   le nombre total de faces GAUCHE BAS DROITE au dessus.
  // - "cell_coord_i * 2"
  //   on avance deux à deux sur les faces d'un même "coté".
  uid[0] = cell_coord_i * 2 + cell_coord_j * (nb_face_x + nb_cell_x);

  // BAS
  // Pour BAS, c'est comme HAUT mais avec un "nombre de face du dessus" en plus.
  uid[2] = uid[0] + (nb_face_x + nb_cell_x);
  // GAUCHE
  // Pour GAUCHE, c'est l'UID de BAS -1.
  uid[3] = uid[2] - 1;
  // DROITE
  // Pour DROITE, c'est l'UID de BAS +1.
  uid[1] = uid[2] + 1;

  uid[0] += first_face_uid;
  uid[1] += first_face_uid;
  uid[2] += first_face_uid;
  uid[3] += first_face_uid;
}

void CartesianMeshNumberingMng::
getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid)
{
  const Int64 cell_coord_i = uidToCoordX(cell_uid, level);
  const Int64 cell_coord_j = uidToCoordY(cell_uid, level);

  if (m_mesh->dimension() == 2) {
    getFaceUids(uid, level, cell_coord_i, cell_coord_j);
  }
  else {
    const Int64 cell_coord_k = uidToCoordZ(cell_uid, level);
    getFaceUids(uid, level, cell_coord_i, cell_coord_j, cell_coord_k);
  }
}

void CartesianMeshNumberingMng::
getCellUidsAround(ArrayView<Int64> uid, Cell cell)
{
  getCellUidsAround(uid, cell.uniqueId(), cell.level());
}

void CartesianMeshNumberingMng::
getCellUidsAround(ArrayView<Int64> uid, Int64 cell_uid, Int32 level)
{
  uid.fill(-1);

  Int64 coord_cell_x = uidToCoordX(cell_uid, level);
  Int64 coord_cell_y = uidToCoordY(cell_uid, level);

  Int64 nb_cells_x = getGlobalNbCellsX(level);
  Int64 nb_cells_y = getGlobalNbCellsY(level);

  if(m_mesh->dimension() == 2){
    ARCANE_ASSERT((uid.size() == 9), ("Size of uid array != 9"));

    for(Integer j = -1; j < 2; ++j){
      Int64 coord_around_cell_y = coord_cell_y + j;
      if(coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y){

        for(Integer i = -1; i < 2; ++i){
          Int64 coord_around_cell_x = coord_cell_x + i;
          if(coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
            uid[(i+1) + ((j+1) * 3)] = getCellUid(level, coord_around_cell_x, coord_around_cell_y);
          }
        }
      }
    }
  }

  else {
    ARCANE_ASSERT((uid.size() == 27), ("Size of uid array != 27"));

    Int64 coord_cell_z = uidToCoordZ(cell_uid, level);
    Int64 nb_cells_z = getGlobalNbCellsZ(level);

    for(Integer k = -1; k < 2; ++k){
      Int64 coord_around_cell_z = coord_cell_z + k;
      if(coord_around_cell_z >= 0 && coord_around_cell_z < nb_cells_z) {

        for(Integer j = -1; j < 2; ++j){
          Int64 coord_around_cell_y = coord_cell_y + j;
          if(coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y){

            for(Integer i = -1; i < 2; ++i){
              Int64 coord_around_cell_x = coord_cell_x + i;
              if(coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
                uid[(i+1) + ((j+1) * 3) + ((k+1) * 9)] = getCellUid(level, coord_around_cell_x, coord_around_cell_y, coord_around_cell_z);
              }
            }
          }
        }
      }
    }
  }
}

void CartesianMeshNumberingMng::
setChildNodeCoordinates(Cell parent_cell)
{
  if (!(parent_cell.itemBase().flags() & ItemFlags::II_JustRefined)) {
    ARCANE_FATAL("Cell not II_JustRefined");
  }

  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();
  Int32 nb_children = parent_cell.nbHChildren();

  auto cell_at_pos = [&](Int32 x, Int32 y, Int32 z, Int64 uid) {
    Cell fin = parent_cell.hChild(x + (y * m_pattern) + (z * m_pattern * m_pattern));
    if (fin.uniqueId() != uid) {
      for (Integer i = 0; i < nb_children; ++i) {
        if (parent_cell.hChild(i).uniqueId() == uid) {
          return parent_cell.hChild(i);
        }
      }
      ARCANE_FATAL("Unknown cell uid -- uid : {0} -- parent_uid : {1}", uid, parent_cell.uniqueId());
    }
    return fin;
  };

  if (m_mesh->dimension() == 2) {
    const Real3& node0(nodes_coords[parent_cell.node(0)]);
    const Real3& node1(nodes_coords[parent_cell.node(1)]);
    const Real3& node2(nodes_coords[parent_cell.node(2)]);
    const Real3& node3(nodes_coords[parent_cell.node(3)]);

    /*
                                        =
                 ┌─────────────────────►= y3
                 │                      = ▲ l
                 ▼                      = ▼
                 X───────────────X◄────►= y2
                /▲              /▲      =
               / │             / │      =
              /  │            /  │      =
             /   │           /   │      =
            /    │          /    │      =
           /     │         /     │      =
          /      │        /      │      =
         /       │       /       │      =
        X───────────────X◄───────│─────►= y1
        ▲        │      ▲        │      = ▲ k
        │        │      │        │      = ▼
        ├──────────────────────────────►= y0
        │        │      │        │      =
        │        │      │        │      =
        ▼        ▼      ▼        ▼
      ==============================
        x0 ◄───► x3     x1 ◄───► x2
             i               j
     */

    auto txty = [&](Integer pos_x, Integer pos_y) -> Real3 {
      const Real x = (Real)pos_x / (Real)m_pattern;
      const Real y = (Real)pos_y / (Real)m_pattern;

      const Real i = (node3.x - node0.x) * y + node0.x;
      const Real j = (node2.x - node1.x) * y + node1.x;

      const Real k = (node1.y - node0.y) * x + node0.y;
      const Real l = (node2.y - node3.y) * x + node3.y;

      const Real tx = (j - i) * x + i;
      const Real ty = (l - k) * y + k;

      //      info() << "[txty]"
      //             << " x : " << x
      //             << " -- y : " << y
      //             << " -- node0 : " << node0
      //             << " -- node1 : " << node1
      //             << " -- node2 : " << node2
      //             << " -- node3 : " << node3
      //             << " -- i : " << i
      //             << " -- j : " << j
      //             << " -- k : " << k
      //             << " -- l : " << l
      //             << " -- tx : " << tx
      //             << " -- ty : " << ty;

      return { tx, ty, 0 };
    };

    const Integer node_1d_2d_x[] = { 0, 1, 1, 0 };
    const Integer node_1d_2d_y[] = { 0, 0, 1, 1 };

    for (Integer j = 0; j < m_pattern; ++j) {
      for (Integer i = 0; i < m_pattern; ++i) {

        Integer begin = (i == 0 && j == 0 ? 0 : j == 0 ? 1
                                                       : 2);
        Integer end = (i == 0 ? getNbNode() : getNbNode() - 1);
        Cell child = cell_at_pos(i, j, 0, getChildCellUidOfCell(parent_cell, i, j));

        for (Integer inode = begin; inode < end; ++inode) {
          nodes_coords[child.node(inode)] = txty(i + node_1d_2d_x[inode], j + node_1d_2d_y[inode]);
          //          Real3 pos = txty(i + node_1d_2d_x[inode], j + node_1d_2d_y[inode]);
          //          nodes_coords[child.node(inode)] = pos;
          //          info() << "Node uid : " << child.node(inode).uniqueId()
          //                 << " -- nodeX : " << (i + node_1d_2d_x[inode])
          //                 << " -- nodeY : " << (j + node_1d_2d_y[inode])
          //                 << " -- Pos : " << pos;
        }
      }
    }
  }

  else {
    const Real3& node0(nodes_coords[parent_cell.node(0)]);
    const Real3& node1(nodes_coords[parent_cell.node(1)]);
    const Real3& node2(nodes_coords[parent_cell.node(2)]);
    const Real3& node3(nodes_coords[parent_cell.node(3)]);
    const Real3& node4(nodes_coords[parent_cell.node(4)]);
    const Real3& node5(nodes_coords[parent_cell.node(5)]);
    const Real3& node6(nodes_coords[parent_cell.node(6)]);
    const Real3& node7(nodes_coords[parent_cell.node(7)]);

    auto txtytz = [&](Integer pos_x, Integer pos_y, Integer pos_z) -> Real3 {
      const Real x = (Real)pos_x / (Real)m_pattern;
      const Real y = (Real)pos_y / (Real)m_pattern;
      const Real z = (Real)pos_z / (Real)m_pattern;

      // Face (m, n, o, p) entre les faces (node0, node1, node2, node3) et (node4, node5, node6, node7).
      const Real3 m = (node4 - node0) * z + node0;
      const Real3 n = (node5 - node1) * z + node1;
      const Real3 o = (node6 - node2) * z + node2;
      const Real3 p = (node7 - node3) * z + node3;

      const Real i = (p.x - m.x) * y + m.x;
      const Real j = (o.x - n.x) * y + n.x;

      const Real tx = (j - i) * x + i;

      const Real k = (n.y - m.y) * x + m.y;
      const Real l = (o.y - p.y) * x + p.y;

      const Real ty = (l - k) * y + k;

      const Real q = (p.z - m.z) * y + m.z;
      const Real r = (o.z - n.z) * y + n.z;

      const Real s = (n.z - m.z) * x + m.z;
      const Real t = (o.z - p.z) * x + p.z;

      const Real tz = (((r - q) * x + q) + ((t - s) * y + s)) * 0.5;

      //      info() << "[txtytz]"
      //             << " x : " << x
      //             << " -- y : " << y
      //             << " -- z : " << z
      //             << " -- node0 : " << node0
      //             << " -- node1 : " << node1
      //             << " -- node2 : " << node2
      //             << " -- node3 : " << node3
      //             << " -- node4 : " << node4
      //             << " -- node5 : " << node5
      //             << " -- node6 : " << node6
      //             << " -- node7 : " << node7
      //             << " -- m : " << m
      //             << " -- n : " << n
      //             << " -- o : " << o
      //             << " -- p : " << p
      //             << " -- j : " << j
      //             << " -- k : " << k
      //             << " -- l : " << l
      //             << " -- q : " << q
      //             << " -- r : " << r
      //             << " -- s : " << s
      //             << " -- t : " << t
      //             << " -- tx : " << tx
      //             << " -- ty : " << ty
      //             << " -- tz : " << tz;

      return { tx, ty, tz };
    };

    const Integer node_1d_3d_x[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    const Integer node_1d_3d_y[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    const Integer node_1d_3d_z[] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    for (Integer k = 0; k < m_pattern; ++k) {
      for (Integer j = 0; j < m_pattern; ++j) {
        for (Integer i = 0; i < m_pattern; ++i) {

          // TODO : éviter les multiples appels pour un même noeud.
          Integer begin = 0;
          Integer end = getNbNode();
          Cell child = cell_at_pos(i, j, k, getChildCellUidOfCell(parent_cell, i, j, k));

          for (Integer inode = begin; inode < end; ++inode) {
            nodes_coords[child.node(inode)] = txtytz(i + node_1d_3d_x[inode], j + node_1d_3d_y[inode], k + node_1d_3d_z[inode]);
            //            Real3 pos = txtytz(i + node_1d_3d_x[inode], j + node_1d_3d_y[inode], k + node_1d_3d_z[inode]);
            //            nodes_coords[child.node(inode)] = pos;
            //            info() << "Node uid : " << child.node(inode).uniqueId()
            //                   << " -- nodeX : " << (i + node_1d_3d_x[inode])
            //                   << " -- nodeY : " << (j + node_1d_3d_y[inode])
            //                   << " -- nodeZ : " << (k + node_1d_3d_z[inode])
            //                   << " -- Pos : " << pos;
          }
        }
      }
    }
  }
}

void CartesianMeshNumberingMng::
setParentNodeCoordinates(Cell parent_cell)
{
  if (!(parent_cell.itemBase().flags() & ItemFlags::II_JustAdded)) {
    ARCANE_FATAL("Cell not II_JustAdded");
  }

  Int32 nb_children = parent_cell.nbHChildren();

  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();

  auto cell_at_pos = [&](Int32 x, Int32 y, Int32 z, Int64 uid) {
    Cell fin = parent_cell.hChild(x + (y * m_pattern) + (z * m_pattern * m_pattern));
    if (fin.uniqueId() != uid) {
      for (Integer i = 0; i < nb_children; ++i) {
        if (parent_cell.hChild(i).uniqueId() == uid) {
          return parent_cell.hChild(i);
        }
      }
      ARCANE_FATAL("Unknown cell uid -- uid : {0} -- parent_uid : {1}", uid, parent_cell.uniqueId());
    }
    return fin;
  };

  if (m_mesh->dimension() == 2) {
    nodes_coords[parent_cell.node(0)] = nodes_coords[cell_at_pos(0, 0, 0, getChildCellUidOfCell(parent_cell, 0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[cell_at_pos(m_pattern - 1, 0, 0, getChildCellUidOfCell(parent_cell, m_pattern - 1, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[cell_at_pos(m_pattern - 1, m_pattern - 1, 0, getChildCellUidOfCell(parent_cell, m_pattern - 1, m_pattern - 1)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[cell_at_pos(0, m_pattern - 1, 0, getChildCellUidOfCell(parent_cell, 0, m_pattern - 1)).node(3)];
  }

  else {

    nodes_coords[parent_cell.node(0)] = nodes_coords[cell_at_pos(0, 0, 0, getChildCellUidOfCell(parent_cell, 0, 0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[cell_at_pos(m_pattern - 1, 0, 0, getChildCellUidOfCell(parent_cell, m_pattern - 1, 0, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[cell_at_pos(m_pattern - 1, m_pattern - 1, 0, getChildCellUidOfCell(parent_cell, m_pattern - 1, m_pattern - 1, 0)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[cell_at_pos(0, m_pattern - 1, 0, getChildCellUidOfCell(parent_cell, 0, m_pattern - 1, 0)).node(3)];

    nodes_coords[parent_cell.node(4)] = nodes_coords[cell_at_pos(0, 0, m_pattern - 1, getChildCellUidOfCell(parent_cell, 0, 0, m_pattern - 1)).node(4)];
    nodes_coords[parent_cell.node(5)] = nodes_coords[cell_at_pos(m_pattern - 1, 0, m_pattern - 1, getChildCellUidOfCell(parent_cell, m_pattern - 1, 0, m_pattern - 1)).node(5)];
    nodes_coords[parent_cell.node(6)] = nodes_coords[cell_at_pos(m_pattern - 1, m_pattern - 1, m_pattern - 1, getChildCellUidOfCell(parent_cell, m_pattern - 1, m_pattern - 1, m_pattern - 1)).node(6)];
    nodes_coords[parent_cell.node(7)] = nodes_coords[cell_at_pos(0, m_pattern - 1, m_pattern - 1, getChildCellUidOfCell(parent_cell, 0, m_pattern - 1, m_pattern - 1)).node(7)];
  }
}

Int64 CartesianMeshNumberingMng::
getParentCellUidOfCell(Cell cell)
{
  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();
  if (m_mesh->dimension() == 2) {
    return getCellUid(level - 1, getOffsetLevelToLevel(uidToCoordX(uid, level), level, level - 1), getOffsetLevelToLevel(uidToCoordY(uid, level), level, level - 1));
  }
  else {
    return getCellUid(level - 1, getOffsetLevelToLevel(uidToCoordX(uid, level), level, level - 1), getOffsetLevelToLevel(uidToCoordY(uid, level), level, level - 1), getOffsetLevelToLevel(uidToCoordZ(uid, level), level, level - 1));
  }
}

Int64 CartesianMeshNumberingMng::
getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent)
{
  ARCANE_ASSERT((child_coord_x_in_parent < m_pattern && child_coord_x_in_parent >= 0), ("Bad child_coord_x_in_parent"))
  ARCANE_ASSERT((child_coord_y_in_parent < m_pattern && child_coord_y_in_parent >= 0), ("Bad child_coord_y_in_parent"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return getCellUid(level + 1,
                    getOffsetLevelToLevel(uidToCoordX(uid, level), level, level + 1) + child_coord_x_in_parent,
                    getOffsetLevelToLevel(uidToCoordY(uid, level), level, level + 1) + child_coord_y_in_parent);
}

Int64 CartesianMeshNumberingMng::
getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent, Int64 child_coord_z_in_parent)
{
  ARCANE_ASSERT((child_coord_x_in_parent < m_pattern && child_coord_x_in_parent >= 0), ("Bad child_coord_x_in_parent"))
  ARCANE_ASSERT((child_coord_y_in_parent < m_pattern && child_coord_y_in_parent >= 0), ("Bad child_coord_y_in_parent"))
  ARCANE_ASSERT((child_coord_z_in_parent < m_pattern && child_coord_z_in_parent >= 0), ("Bad child_coord_z_in_parent"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return getCellUid(level + 1, getOffsetLevelToLevel(uidToCoordX(uid, level), level, level + 1) + child_coord_x_in_parent, getOffsetLevelToLevel(uidToCoordY(uid, level), level, level + 1) + child_coord_y_in_parent, getOffsetLevelToLevel(uidToCoordZ(uid, level), level, level + 1) + child_coord_z_in_parent);
}

Int64 CartesianMeshNumberingMng::
getChildCellUidOfCell(Cell cell, Int64 child_index_in_parent)
{
  if (m_mesh->dimension() == 2) {
    ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

    const Int64 coord_x = child_index_in_parent % m_pattern;
    const Int64 coord_y = child_index_in_parent / m_pattern;

    return getChildCellUidOfCell(cell, coord_x, coord_y);
  }
  else {
    ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

    const Int64 to_2d = child_index_in_parent % (m_pattern * m_pattern);
    const Int64 coord_x = to_2d % m_pattern;
    const Int64 coord_y = to_2d / m_pattern;
    const Int64 coord_z = child_index_in_parent / (m_pattern * m_pattern);

    return getChildCellUidOfCell(cell, coord_x, coord_y, coord_z);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
