// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.cc                                (C) 2000-2023 */
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

  m_first_cell_uid_level.add(0);
  m_first_node_uid_level.add(0);
  m_first_face_uid_level.add(0);
}

Int64 CartesianMeshNumberingMng::
getFirstCellUidLevel(Integer level)
{
  if(level < m_first_cell_uid_level.size()){
   return m_first_cell_uid_level[level];
  }

  const Int32 dimension = m_mesh->dimension();

  Int64 level_i_nb_cell_x = getGlobalNbCellsX(m_nb_cell_level.size());
  Int64 level_i_nb_cell_y = getGlobalNbCellsY(m_nb_cell_level.size());
  Int64 level_i_nb_cell_z = getGlobalNbCellsZ(m_nb_cell_level.size());

  for(Integer i = m_nb_cell_level.size(); i < level+1; ++i){
    if(dimension == 2){
      m_nb_cell_level.add(level_i_nb_cell_x * level_i_nb_cell_y);
    }
    else if(dimension == 3){
      m_nb_cell_level.add(level_i_nb_cell_x * level_i_nb_cell_y * level_i_nb_cell_z);
    }
    m_first_cell_uid_level.add(m_first_cell_uid_level[i] + m_nb_cell_level[i]);

    level_i_nb_cell_x *= m_pattern;
    level_i_nb_cell_y *= m_pattern;
    level_i_nb_cell_z *= m_pattern;
  }

  return m_first_cell_uid_level[level];
}


Int64 CartesianMeshNumberingMng::
getFirstNodeUidLevel(Integer level)
{
  if(level < m_first_node_uid_level.size()){
    return m_first_node_uid_level[level];
  }

  const Int32 dimension = m_mesh->dimension();

  Int64 level_i_nb_cell_x = getGlobalNbCellsX(m_nb_node_level.size());
  Int64 level_i_nb_cell_y = getGlobalNbCellsY(m_nb_node_level.size());
  Int64 level_i_nb_cell_z = getGlobalNbCellsZ(m_nb_node_level.size());

  for(Integer i = m_nb_node_level.size(); i < level+1; ++i){
    if(dimension == 2){
      m_nb_node_level.add((level_i_nb_cell_x + 1) * (level_i_nb_cell_y + 1));
    }
    else if(dimension == 3){
      m_nb_node_level.add((level_i_nb_cell_x + 1) * (level_i_nb_cell_y + 1) * (level_i_nb_cell_z + 1));
    }
    m_first_node_uid_level.add(m_first_node_uid_level[i] + m_nb_node_level[i]);

    level_i_nb_cell_x *= m_pattern;
    level_i_nb_cell_y *= m_pattern;
    level_i_nb_cell_z *= m_pattern;
  }

  return m_first_node_uid_level[level];
}


Int64 CartesianMeshNumberingMng::
getFirstFaceUidLevel(Integer level)
{
  if(level < m_first_face_uid_level.size()){
    return m_first_face_uid_level[level];
  }

  const Int32 dimension = m_mesh->dimension();

  Int64 level_i_nb_cell_x = getGlobalNbCellsX(m_nb_face_level.size());
  Int64 level_i_nb_cell_y = getGlobalNbCellsY(m_nb_face_level.size());
  Int64 level_i_nb_cell_z = getGlobalNbCellsZ(m_nb_face_level.size());

  for(Integer i = m_nb_face_level.size(); i < level+1; ++i){
    if(dimension == 2){
      m_nb_face_level.add((level_i_nb_cell_x * level_i_nb_cell_y) * 2 + level_i_nb_cell_x*2 + level_i_nb_cell_y);
    }
    else if(dimension == 3){
      m_nb_face_level.add((level_i_nb_cell_z + 1) * level_i_nb_cell_x * level_i_nb_cell_y
                        + (level_i_nb_cell_x + 1) * level_i_nb_cell_y * level_i_nb_cell_z
                        + (level_i_nb_cell_y + 1) * level_i_nb_cell_z * level_i_nb_cell_x);
    }
    m_first_face_uid_level.add(m_first_face_uid_level[i] + m_nb_face_level[i]);

    level_i_nb_cell_x *= m_pattern;
    level_i_nb_cell_y *= m_pattern;
    level_i_nb_cell_z *= m_pattern;
  }

  return m_first_face_uid_level[level];
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsX(Integer level) const
{
  return m_nb_cell_x * static_cast<Int64>(std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsY(Integer level) const
{
  return m_nb_cell_y * static_cast<Int64>(std::pow(m_pattern, level));
}

Int64 CartesianMeshNumberingMng::
getGlobalNbCellsZ(Integer level) const
{
  return m_nb_cell_z * static_cast<Int64>(std::pow(m_pattern, level));
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
  ARCANE_ASSERT((level_from < level_to), ("Pb level_from level_to"));
  return coord * m_pattern * (level_to - level_from);
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
getCellUidsAround(ArrayView<Int64> uid, Cell cell)
{
  uid.fill(-1);

  Int64 coord_cell_x = uidToCoordX(cell);
  Int64 coord_cell_y = uidToCoordY(cell);

  Integer level = cell.level();
  Int64 nb_cells_y = getGlobalNbCellsY(level);
  Int64 nb_cells_x = getGlobalNbCellsX(level);

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

    Int64 coord_cell_z = uidToCoordZ(cell);
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
setNodeCoordinates(Cell child_cell)
{
  if (!(child_cell.itemBase().flags() & ItemFlags::II_JustAdded)) {
    ARCANE_FATAL("Cell not II_JustAdded");
  }
  Cell parent_cell = child_cell.hParent();

  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();

  Real3 pos(
    Real(uidToCoordX(child_cell) % m_pattern),
    Real(uidToCoordY(child_cell) % m_pattern),
    (m_mesh->dimension() == 2 ? 0 : Real(uidToCoordZ(child_cell) % m_pattern))
  );

  Real3 size_child_cell;

  if(m_mesh->dimension() == 2) {
    size_child_cell = nodes_coords[parent_cell.node(2)] - nodes_coords[parent_cell.node(0)];
  }
  else{
    size_child_cell = nodes_coords[parent_cell.node(6)] - nodes_coords[parent_cell.node(0)];
  }
  size_child_cell /= m_pattern;

  Real3 origin_parent_cell(nodes_coords[parent_cell.node(0)]);
  Real3 origin_child_cell(origin_parent_cell + (size_child_cell * pos));

  nodes_coords[child_cell.node(0)] = origin_child_cell;

  nodes_coords[child_cell.node(1)] = origin_child_cell;
  nodes_coords[child_cell.node(1)].x += size_child_cell.x;

  nodes_coords[child_cell.node(2)] = origin_child_cell;
  nodes_coords[child_cell.node(2)].x += size_child_cell.x;
  nodes_coords[child_cell.node(2)].y += size_child_cell.y;

  nodes_coords[child_cell.node(3)] = origin_child_cell;
  nodes_coords[child_cell.node(3)].y += size_child_cell.y;

  if(m_mesh->dimension() == 3) {
    nodes_coords[child_cell.node(4)] = origin_child_cell;
    nodes_coords[child_cell.node(4)].z += size_child_cell.z;

    nodes_coords[child_cell.node(5)] = origin_child_cell;
    nodes_coords[child_cell.node(5)].x += size_child_cell.x;
    nodes_coords[child_cell.node(5)].z += size_child_cell.z;

    nodes_coords[child_cell.node(6)] = origin_child_cell + size_child_cell;

    nodes_coords[child_cell.node(7)] = origin_child_cell;
    nodes_coords[child_cell.node(7)].y += size_child_cell.y;
    nodes_coords[child_cell.node(7)].z += size_child_cell.z;
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
