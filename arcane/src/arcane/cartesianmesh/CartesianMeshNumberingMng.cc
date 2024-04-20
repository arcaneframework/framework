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
, m_dimension(mesh->dimension())
, m_pattern(2)
, m_max_level(0)
, m_min_level(0)
{
  auto* m_generation_info = ICartesianMeshGenerationInfo::getReference(m_mesh,true);

  Int64ConstArrayView global_nb_cells_by_direction = m_generation_info->globalNbCells();
  m_nb_cell.x = global_nb_cells_by_direction[MD_DirX];
  m_nb_cell.y = global_nb_cells_by_direction[MD_DirY];
  m_nb_cell.z = ((m_dimension == 2) ? 1 : global_nb_cells_by_direction[MD_DirZ]);

  if (m_nb_cell.x <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", m_nb_cell.x);
  if (m_nb_cell.y <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", m_nb_cell.y);
  if (m_nb_cell.z <= 0)
    ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", m_nb_cell.z);

  m_p_to_l_level.add(0);

  if (m_dimension == 2) {
    m_latest_cell_uid = m_nb_cell.x * m_nb_cell.y;
    m_latest_node_uid = (m_nb_cell.x + 1) * (m_nb_cell.y + 1);
    m_latest_face_uid = (m_nb_cell.x * m_nb_cell.y) * 2 + m_nb_cell.x * 2 + m_nb_cell.y;
  }
  else {
    m_latest_cell_uid = m_nb_cell.x * m_nb_cell.y * m_nb_cell.z;
    m_latest_node_uid = (m_nb_cell.x + 1) * (m_nb_cell.y + 1) * (m_nb_cell.z + 1);
    m_latest_face_uid = (m_nb_cell.z + 1) * m_nb_cell.x * m_nb_cell.y + (m_nb_cell.x + 1) * m_nb_cell.y * m_nb_cell.z + (m_nb_cell.y + 1) * m_nb_cell.z * m_nb_cell.x;
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

  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));

  if (m_dimension == 2) {
    m_latest_cell_uid += nb_cell.x * nb_cell.y;
    m_latest_node_uid += (nb_cell.x + 1) * (nb_cell.y + 1);
    m_latest_face_uid += (nb_cell.x * nb_cell.y) * 2 + nb_cell.x * 2 + nb_cell.y;
  }
  else {
    m_latest_cell_uid += nb_cell.x * nb_cell.y * nb_cell.z;
    m_latest_node_uid += (nb_cell.x + 1) * (nb_cell.y + 1) * (nb_cell.z + 1);
    m_latest_face_uid += (nb_cell.z + 1) * nb_cell.x * nb_cell.y + (nb_cell.x + 1) * nb_cell.y * nb_cell.z + (nb_cell.y + 1) * nb_cell.z * nb_cell.x;
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

  m_nb_cell /= (m_pattern * nb_levels_to_add);

  // ----------
  // CartesianMeshCoarsening2::_recomputeMeshGenerationInfo()
  // Recalcule les informations sur le nombre de mailles par direction.
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(m_mesh, false);
  if (!cmgi)
    return;

  {
    ConstArrayView<Int64> v = cmgi->ownCellOffsets();
    cmgi->setOwnCellOffsets(v[0] / m_pattern, v[1] / m_pattern, v[2] / m_pattern);
  }
  {
    ConstArrayView<Int64> v = cmgi->globalNbCells();
    cmgi->setGlobalNbCells(v[0] / m_pattern, v[1] / m_pattern, v[2] / m_pattern);
  }
  {
    ConstArrayView<Int32> v = cmgi->ownNbCells();
    cmgi->setOwnNbCells(v[0] / m_pattern, v[1] / m_pattern, v[2] / m_pattern);
  }
  cmgi->setFirstOwnCellUniqueId(firstCellUniqueId(0));
  // CartesianMeshCoarsening2::_recomputeMeshGenerationInfo()
  // ----------
}

Int64 CartesianMeshNumberingMng::
firstCellUniqueId(Integer level)
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
firstNodeUniqueId(Integer level)
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
firstFaceUniqueId(Integer level)
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
globalNbCellsX(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell.x) * std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
globalNbCellsY(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell.y) * std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
globalNbCellsZ(Integer level) const
{
  return static_cast<Int64>(static_cast<Real>(m_nb_cell.z) * std::pow(m_pattern, level));
}

Integer CartesianMeshNumberingMng::
pattern() const
{
  return m_pattern;
}

// Tant que l'on a un unique "pattern" pour x, y, z, pas besoin de trois méthodes.
Int64 CartesianMeshNumberingMng::
offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const
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
cellUniqueIdToCoordX(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return to2d % nb_cell_x;
}

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Cell cell)
{
  return cellUniqueIdToCoordX(cell.uniqueId(), cell.level());
}

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return to2d / nb_cell_x;
}

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Cell cell)
{
  return cellUniqueIdToCoordY(cell.uniqueId(), cell.level());
}

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Int64 uid, Integer level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  return uid / (nb_cell_x * nb_cell_y);
}

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Cell cell)
{
  return cellUniqueIdToCoordZ(cell.uniqueId(), cell.level());
}


Int64 CartesianMeshNumberingMng::
cellUniqueId(Integer level, Int64x3 cell_coord)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  return (cell_coord.x + cell_coord.y * nb_cell_x + cell_coord.z * nb_cell_x * nb_cell_y) + first_cell_uid;
}

Int64 CartesianMeshNumberingMng::
cellUniqueId(Integer level, Int64x2 cell_coord)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  return (cell_coord.x + cell_coord.y * nb_cell_x) + first_cell_uid;
}



Integer CartesianMeshNumberingMng::
nbNodeByCell()
{
  return static_cast<Integer>(std::pow(m_pattern, m_mesh->dimension()));
}

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord)
{
  if (uid.size() != nbNodeByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = globalNbCellsX(level) + 1;
  const Int64 nb_node_y = globalNbCellsY(level) + 1;
  const Int64 first_node_uid = firstNodeUniqueId(level);

  uid[0] = (cell_coord.x + 0) + ((cell_coord.y + 0) * nb_node_x) + ((cell_coord.z + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[1] = (cell_coord.x + 1) + ((cell_coord.y + 0) * nb_node_x) + ((cell_coord.z + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[2] = (cell_coord.x + 1) + ((cell_coord.y + 1) * nb_node_x) + ((cell_coord.z + 0) * nb_node_x * nb_node_y) + first_node_uid;
  uid[3] = (cell_coord.x + 0) + ((cell_coord.y + 1) * nb_node_x) + ((cell_coord.z + 0) * nb_node_x * nb_node_y) + first_node_uid;

  uid[4] = (cell_coord.x + 0) + ((cell_coord.y + 0) * nb_node_x) + ((cell_coord.z + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[5] = (cell_coord.x + 1) + ((cell_coord.y + 0) * nb_node_x) + ((cell_coord.z + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[6] = (cell_coord.x + 1) + ((cell_coord.y + 1) * nb_node_x) + ((cell_coord.z + 1) * nb_node_x * nb_node_y) + first_node_uid;
  uid[7] = (cell_coord.x + 0) + ((cell_coord.y + 1) * nb_node_x) + ((cell_coord.z + 1) * nb_node_x * nb_node_y) + first_node_uid;
}

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord)
{
  if (uid.size() != nbNodeByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = globalNbCellsX(level) + 1;
  const Int64 first_node_uid = firstNodeUniqueId(level);

  uid[0] = (cell_coord.x + 0) + ((cell_coord.y + 0) * nb_node_x) + first_node_uid;
  uid[1] = (cell_coord.x + 1) + ((cell_coord.y + 0) * nb_node_x) + first_node_uid;
  uid[2] = (cell_coord.x + 1) + ((cell_coord.y + 1) * nb_node_x) + first_node_uid;
  uid[3] = (cell_coord.x + 0) + ((cell_coord.y + 1) * nb_node_x) + first_node_uid;
}

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid)
{
  if (m_dimension == 2) {
    const Int64x2 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level));
    cellNodeUniqueIds(uid, level, cell_coord);
  }
  else {
    const Int64x3 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level), cellUniqueIdToCoordZ(cell_uid, level));
    cellNodeUniqueIds(uid, level, cell_coord);
  }
}

Integer CartesianMeshNumberingMng::
nbFaceByCell()
{
  return m_pattern * m_dimension;
}

void CartesianMeshNumberingMng::
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord)
{
  if (uid.size() != nbFaceByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));
  const Int64x3 nb_face(nb_cell + 1);

  const Int64 first_face_uid = firstFaceUniqueId(level);

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

  const Int64 total_face_xy = nb_face.z * nb_cell.x * nb_cell.y;
  const Int64 total_face_xy_yz = total_face_xy + nb_face.x * nb_cell.y * nb_cell.z;

  const Int64 nb_cell_before_j = cell_coord.y * nb_cell.x;

  uid[0] = (cell_coord.z * nb_cell.x * nb_cell.y) + nb_cell_before_j + (cell_coord.x);

  uid[3] = uid[0] + nb_cell.x * nb_cell.y;

  uid[1] = (cell_coord.z * nb_face.x * nb_cell.y) + (cell_coord.y * nb_face.x) + (cell_coord.x) + total_face_xy;

  uid[4] = uid[1] + 1;

  uid[2] = (cell_coord.z * nb_cell.x * nb_face.y) + nb_cell_before_j + (cell_coord.x) + total_face_xy_yz;

  uid[5] = uid[2] + nb_cell.x;

  uid[0] += first_face_uid;
  uid[1] += first_face_uid;
  uid[2] += first_face_uid;
  uid[3] += first_face_uid;
  uid[4] += first_face_uid;
  uid[5] += first_face_uid;
}


void CartesianMeshNumberingMng::
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord)
{
  if (uid.size() != nbFaceByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 first_face_uid = firstFaceUniqueId(level);

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
  // - "cell_coord.y * (current_level_nb_face_x + current_level_nb_cell_x)" :
  //   le nombre total de faces GAUCHE BAS DROITE au dessus.
  // - "cell_coord.x * 2"
  //   on avance deux à deux sur les faces d'un même "coté".
  uid[0] = cell_coord.x * 2 + cell_coord.y * (nb_face_x + nb_cell_x);

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
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid)
{
  if (m_dimension == 2) {
    const Int64x2 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level));
    cellFaceUniqueIds(uid, level, cell_coord);
  }
  else {
    const Int64x3 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level), cellUniqueIdToCoordZ(cell_uid, level));
    cellFaceUniqueIds(uid, level, cell_coord);
  }
}

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell)
{
  cellUniqueIdsAroundCell(uid, cell.uniqueId(), cell.level());
}

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level)
{
  uid.fill(-1);

  const Int64 coord_cell_x = cellUniqueIdToCoordX(cell_uid, level);
  const Int64 coord_cell_y = cellUniqueIdToCoordY(cell_uid, level);

  const Int64 nb_cells_x = globalNbCellsX(level);
  const Int64 nb_cells_y = globalNbCellsY(level);

  if (m_dimension == 2) {
    ARCANE_ASSERT((uid.size() == 9), ("Size of uid array != 9"));

    for(Integer j = -1; j < 2; ++j){
      const Int64 coord_around_cell_y = coord_cell_y + j;
      if(coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y){

        for(Integer i = -1; i < 2; ++i){
          const Int64 coord_around_cell_x = coord_cell_x + i;
          if(coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
            uid[(i + 1) + ((j + 1) * 3)] = cellUniqueId(level, Int64x2(coord_around_cell_x, coord_around_cell_y));
          }
        }
      }
    }
  }

  else {
    ARCANE_ASSERT((uid.size() == 27), ("Size of uid array != 27"));

    const Int64 coord_cell_z = cellUniqueIdToCoordZ(cell_uid, level);
    const Int64 nb_cells_z = globalNbCellsZ(level);

    for(Integer k = -1; k < 2; ++k){
      const Int64 coord_around_cell_z = coord_cell_z + k;
      if(coord_around_cell_z >= 0 && coord_around_cell_z < nb_cells_z) {

        for(Integer j = -1; j < 2; ++j){
          const Int64 coord_around_cell_y = coord_cell_y + j;
          if(coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y){

            for(Integer i = -1; i < 2; ++i){
              const Int64 coord_around_cell_x = coord_cell_x + i;
              if(coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
                uid[(i + 1) + ((j + 1) * 3) + ((k + 1) * 9)] = cellUniqueId(level, Int64x3(coord_around_cell_x, coord_around_cell_y, coord_around_cell_z));
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

  const Real3& node0(nodes_coords[parent_cell.node(0)]);
  const Real3& node1(nodes_coords[parent_cell.node(1)]);
  const Real3& node2(nodes_coords[parent_cell.node(2)]);
  const Real3& node3(nodes_coords[parent_cell.node(3)]);

  if (m_dimension == 2) {

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
    /*!
     * \brief Lambda permettant de déterminer la position d'un noeud enfant
     * dans une maille parent.
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

      /*
      info() << "[txty]"
             << " x : " << x
             << " -- y : " << y
             << " -- node0 : " << node0
             << " -- node1 : " << node1
             << " -- node2 : " << node2
             << " -- node3 : " << node3
             << " -- i : " << i
             << " -- j : " << j
             << " -- k : " << k
             << " -- l : " << l
             << " -- tx : " << tx
             << " -- ty : " << ty;
      */
      return { tx, ty, 0 };
    };

    const Integer node_1d_2d_x[] = { 0, 1, 1, 0 };
    const Integer node_1d_2d_y[] = { 0, 0, 1, 1 };

    for (Integer j = 0; j < m_pattern; ++j) {
      for (Integer i = 0; i < m_pattern; ++i) {

        Integer begin = (i == 0 && j == 0 ? 0 : j == 0 ? 1
                                                       : 2);
        Integer end = (i == 0 ? nbNodeByCell() : nbNodeByCell() - 1);
        Cell child = childCellOfCell(parent_cell, Int64x2(i, j));

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
    const Real3& node4(nodes_coords[parent_cell.node(4)]);
    const Real3& node5(nodes_coords[parent_cell.node(5)]);
    const Real3& node6(nodes_coords[parent_cell.node(6)]);
    const Real3& node7(nodes_coords[parent_cell.node(7)]);

    /*!
     * \brief Lambda permettant de déterminer la position d'un noeud enfant
     * dans une maille parent.
     */
    auto txtytz = [&](Integer pos_x, Integer pos_y, Integer pos_z) -> Real3 {
      const Real x = (Real)pos_x / (Real)m_pattern;
      const Real y = (Real)pos_y / (Real)m_pattern;
      const Real z = (Real)pos_z / (Real)m_pattern;

      // Face (m, n, o, p) entre les faces (node0, node1, node2, node3) et (node4, node5, node6, node7).
      const Real3 m = (node4 - node0) * z + node0;
      const Real3 n = (node5 - node1) * z + node1;
      const Real3 o = (node6 - node2) * z + node2;
      const Real3 p = (node7 - node3) * z + node3;

      // On calcule tx et ty comme en 2D mais sur la face (m, n, o, p).
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

      /*
      info() << "[txtytz]"
             << " x : " << x
             << " -- y : " << y
             << " -- z : " << z
             << " -- node0 : " << node0
             << " -- node1 : " << node1
             << " -- node2 : " << node2
             << " -- node3 : " << node3
             << " -- node4 : " << node4
             << " -- node5 : " << node5
             << " -- node6 : " << node6
             << " -- node7 : " << node7
             << " -- m : " << m
             << " -- n : " << n
             << " -- o : " << o
             << " -- p : " << p
             << " -- j : " << j
             << " -- k : " << k
             << " -- l : " << l
             << " -- q : " << q
             << " -- r : " << r
             << " -- s : " << s
             << " -- t : " << t
             << " -- tx : " << tx
             << " -- ty : " << ty
             << " -- tz : " << tz;
      */
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
          Integer end = nbNodeByCell();
          Cell child = childCellOfCell(parent_cell, Int64x3(i, j, k));

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

  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();

  if (m_dimension == 2) {
    nodes_coords[parent_cell.node(0)] = nodes_coords[childCellOfCell(parent_cell, Int64x2(0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[childCellOfCell(parent_cell, Int64x2(m_pattern - 1, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[childCellOfCell(parent_cell, Int64x2(m_pattern - 1, m_pattern - 1)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[childCellOfCell(parent_cell, Int64x2(0, m_pattern - 1)).node(3)];
  }

  else {
    nodes_coords[parent_cell.node(0)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(0, 0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(m_pattern - 1, 0, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(m_pattern - 1, m_pattern - 1, 0)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(0, m_pattern - 1, 0)).node(3)];

    nodes_coords[parent_cell.node(4)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(0, 0, m_pattern - 1)).node(4)];
    nodes_coords[parent_cell.node(5)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(m_pattern - 1, 0, m_pattern - 1)).node(5)];
    nodes_coords[parent_cell.node(6)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(m_pattern - 1, m_pattern - 1, m_pattern - 1)).node(6)];
    nodes_coords[parent_cell.node(7)] = nodes_coords[childCellOfCell(parent_cell, Int64x3(0, m_pattern - 1, m_pattern - 1)).node(7)];
  }
}

Int64 CartesianMeshNumberingMng::
parentCellUniqueIdOfCell(Cell cell)
{
  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();
  if (m_dimension == 2) {
    return cellUniqueId(level - 1,
                        Int64x2(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level - 1),
                                offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level - 1)));
  }
  else {
    return cellUniqueId(level - 1,
                        Int64x3(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level - 1),
                                offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level - 1),
                                offsetLevelToLevel(cellUniqueIdToCoordZ(uid, level), level, level - 1)));
  }
}

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return cellUniqueId(level + 1,
                      Int64x2(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level + 1) + child_coord_in_parent.x,
                              offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level + 1) + child_coord_in_parent.y));
}

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, Int64x2 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  Cell child = cell.hChild((Int32)child_coord_in_parent.x + ((Int32)child_coord_in_parent.y * m_pattern));
  const Int64 uid = childCellUniqueIdOfCell(cell, child_coord_in_parent);

  // Si jamais la maille à l'index calculé ne correspond pas à l'uniqueId
  // recherché, on recherche parmi les autres mailles enfants.
  if (child.uniqueId() != uid) {
    const Int32 nb_children = cell.nbHChildren();
    for (Integer i = 0; i < nb_children; ++i) {
      if (cell.hChild(i).uniqueId() == uid) {
        return cell.hChild(i);
      }
    }
    ARCANE_FATAL("Unknown cell uid -- uid : {0} -- parent_uid : {1}", uid, cell.uniqueId());
  }
  return child;
}

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))
  ARCANE_ASSERT((child_coord_in_parent.z < m_pattern && child_coord_in_parent.z >= 0), ("Bad child_coord_in_parent.z"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return cellUniqueId(level + 1,
                      Int64x3(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level + 1) + child_coord_in_parent.x,
                              offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level + 1) + child_coord_in_parent.y,
                              offsetLevelToLevel(cellUniqueIdToCoordZ(uid, level), level, level + 1) + child_coord_in_parent.z));
}

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, Int64x3 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  Cell child = cell.hChild((Int32)child_coord_in_parent.x + ((Int32)child_coord_in_parent.y * m_pattern) + ((Int32)child_coord_in_parent.z * m_pattern * m_pattern));
  const Int64 uid = childCellUniqueIdOfCell(cell, child_coord_in_parent);

  // Si jamais la maille à l'index calculé ne correspond pas à l'uniqueId
  // recherché, on recherche parmi les autres mailles enfants.
  if (child.uniqueId() != uid) {
    const Int32 nb_children = cell.nbHChildren();
    for (Integer i = 0; i < nb_children; ++i) {
      if (cell.hChild(i).uniqueId() == uid) {
        return cell.hChild(i);
      }
    }
    ARCANE_FATAL("Unknown cell uid -- uid : {0} -- parent_uid : {1}", uid, cell.uniqueId());
  }
  return child;
}

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent)
{
  if (m_dimension == 2) {
    ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

    return childCellUniqueIdOfCell(cell,
                                   Int64x2(
                                   child_index_in_parent % m_pattern,
                                   child_index_in_parent / m_pattern));
  }

  else {
    ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

    const Int64 to_2d = child_index_in_parent % (m_pattern * m_pattern);
    return childCellUniqueIdOfCell(cell,
                                   Int64x3(
                                   to_2d % m_pattern,
                                   to_2d / m_pattern,
                                   child_index_in_parent / (m_pattern * m_pattern)));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
