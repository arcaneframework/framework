// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMngInternal.cc                        (C) 2000-2026 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* des mailles et des noeuds est assez classique, la numérotation des faces  */
/* est expliquée (entre autres) dans les méthodes 'faceUniqueId()' et        */
/* 'cellFaceUniqueIds()'.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianMeshNumberingMngInternal.h"

#include "arcane/utils/Vector2.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/Properties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshNumberingMngInternal::
CartesianMeshNumberingMngInternal(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_dimension(mesh->dimension())
, m_pattern(2)
, m_max_level(0)
, m_min_level(0)
, m_converting_numbering_face(true)
, m_ori_level(0)
{
  {
    const auto* m_generation_info = ICartesianMeshGenerationInfo::getReference(m_mesh, true);
    ConstArrayView<Int64> global_nb_cells_by_direction = m_generation_info->globalNbCells();

    const Int64x3 nb_cell_ground = {
      global_nb_cells_by_direction[MD_DirX],
      global_nb_cells_by_direction[MD_DirY],
      ((m_dimension == 2) ? 1 : global_nb_cells_by_direction[MD_DirZ])
    };

    m_nb_cell_ground.x = static_cast<CartCoord>(nb_cell_ground.x);
    m_nb_cell_ground.y = static_cast<CartCoord>(nb_cell_ground.y);
    m_nb_cell_ground.z = static_cast<CartCoord>(nb_cell_ground.z);

    if (nb_cell_ground.x <= 0)
      ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirX] (should be >0)", nb_cell_ground.x);
    if (nb_cell_ground.y <= 0)
      ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirY] (should be >0)", nb_cell_ground.y);
    if (nb_cell_ground.z <= 0)
      ARCANE_FATAL("Bad value '{0}' for globalNbCells()[MD_DirZ] (should be >0)", nb_cell_ground.z);

    if (m_dimension == 2) {
      m_latest_cell_uid = nb_cell_ground.x * nb_cell_ground.y;
      m_latest_node_uid = (nb_cell_ground.x + 1) * (nb_cell_ground.y + 1);
      m_latest_face_uid = (nb_cell_ground.x * nb_cell_ground.y) * 2 + nb_cell_ground.x * 2 + nb_cell_ground.y;
    }
    else {
      m_latest_cell_uid = nb_cell_ground.x * nb_cell_ground.y * nb_cell_ground.z;
      m_latest_node_uid = (nb_cell_ground.x + 1) * (nb_cell_ground.y + 1) * (nb_cell_ground.z + 1);
      m_latest_face_uid = (nb_cell_ground.x + 1) * nb_cell_ground.y * nb_cell_ground.z + (nb_cell_ground.y + 1) * nb_cell_ground.z * nb_cell_ground.x + (nb_cell_ground.z + 1) * nb_cell_ground.x * nb_cell_ground.y;
    }
  }

  m_first_cell_uid_level.add(0);
  m_first_node_uid_level.add(0);
  m_first_face_uid_level.add(0);

  // Tant qu'on utilise la numérotation d'origine pour le niveau 0, on doit utiliser
  // une conversion de la numérotation d'origine vers la nouvelle.
  // TODO AH : Ça risque de pas très bien se passer en cas de repartitionnement...
  if (m_converting_numbering_face) {
    UniqueArray<Int64> face_uid(CartesianMeshNumberingMngInternal::nbFaceByCell());
    ENUMERATE_ (Cell, icell, m_mesh->allLevelCells(0)) {
      CartesianMeshNumberingMngInternal::cellFaceUniqueIds(icell->uniqueId(), 0, face_uid);
      for (Integer i = 0; i < CartesianMeshNumberingMngInternal::nbFaceByCell(); ++i) {
        m_face_ori_numbering_to_new[icell->face(i).uniqueId()] = face_uid[i];
        m_face_new_numbering_to_ori[face_uid[i]] = icell->face(i).uniqueId();
        //        debug() << "Face Ori <-> New -- Ori : " << icell->face(i).uniqueId() << " -- New : " << face_uid[i];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
build()
{
  m_properties = makeRef(new Properties(*(m_mesh->properties()), "CartesianMeshNumberingMngInternal"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
saveInfosInProperties()
{
  m_properties->set("Version", 1);
  m_properties->set("FirstCellUIDByLevel", m_first_cell_uid_level);

  // Voir pour le recalculer à la reprise.
  m_properties->set("FirstNodeUIDByLevel", m_first_node_uid_level);
  m_properties->set("FirstFaceUIDByLevel", m_first_face_uid_level);

  m_properties->set("OriginalGroundLevelForConverting", m_ori_level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
recreateFromDump()
{
  Int32 v = m_properties->getInt32("Version");
  if (v != 1)
    ARCANE_FATAL("Bad numbering mng version: trying to read from incompatible checkpoint v={0} expected={1}", v, 1);

  m_properties->get("FirstCellUIDByLevel", m_first_cell_uid_level);
  m_properties->get("FirstNodeUIDByLevel", m_first_node_uid_level);
  m_properties->get("FirstFaceUIDByLevel", m_first_face_uid_level);

  m_properties->get("OriginalGroundLevelForConverting", m_ori_level);
  if (m_ori_level == -1) {
    m_converting_numbering_face = false;
    m_face_ori_numbering_to_new.clear();
    m_face_new_numbering_to_ori.clear();
  }

  m_nb_cell_ground = { globalNbCellsX(0), globalNbCellsY(0), globalNbCellsZ(0) };

  m_max_level = m_first_cell_uid_level.size() - 1;

  {
    Integer pos = 0;
    Int64 max = 0;
    Integer iter = 0;
    for (const Int64 elem : m_first_cell_uid_level) {
      if (elem > max) {
        max = elem;
        pos = iter;
      }
      iter++;
    }
    m_latest_cell_uid = m_first_cell_uid_level[pos] + nbCellInLevel(pos);
    m_latest_node_uid = m_first_node_uid_level[pos] + nbNodeInLevel(pos);
    m_latest_face_uid = m_first_face_uid_level[pos] + nbFaceInLevel(pos);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
renumberingFacesLevel0FromOriginalArcaneNumbering()
{
  if (!m_converting_numbering_face)
    return;

  info() << "Renumbering faces of level 0...";

  UniqueArray<Int64> face_uid(nbFaceByCell());
  ENUMERATE_ (Cell, icell, m_mesh->allLevelCells(m_ori_level)) {
    cellFaceUniqueIds(icell->uniqueId(), m_ori_level, face_uid);
    for (Integer i = 0; i < nbFaceByCell(); ++i) {
      // debug() << "Face Ori <-> New -- Ori : " << icell->face(i).uniqueId() << " -- New : " << face_uid[i];
      icell->face(i).mutableItemBase().setUniqueId(face_uid[i]);
    }
  }
  m_mesh->faceFamily()->notifyItemsUniqueIdChanged();
  m_mesh->modifier()->endUpdate();
  m_mesh->checkValidMesh();

  m_converting_numbering_face = false;
  m_ori_level = -1;
  m_face_ori_numbering_to_new.clear();
  m_face_new_numbering_to_ori.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
printStatus()
{
  ITraceMng* tm = m_mesh->traceMng();
  Trace::Setter _(tm, "CartesianMeshNumberingMngInternal");

  tm->info() << "CartesianMeshNumberingMngInternal status :";

  // tm->info() << "FirstCellUIDByLevel : " << m_first_cell_uid_level;
  // tm->info() << "FirstNodeUIDByLevel : " << m_first_node_uid_level;
  // tm->info() << "FirstFaceUIDByLevel : " << m_first_face_uid_level;

  tm->info() << "LatestCellUID : " << m_latest_cell_uid;
  tm->info() << "LatestNodeUID : " << m_latest_node_uid;
  tm->info() << "LatestFaceUID : " << m_latest_face_uid;

  tm->info() << "MinLevel : " << m_min_level;
  tm->info() << "MaxLevel : " << m_max_level;

  tm->info() << "GroundLevelNbCells : " << m_nb_cell_ground;

  if (m_ori_level == -1) {
    tm->info() << "Ground Level is renumbered";
  }
  else {
    tm->info() << "Ground Level is not renumbered -- OriginalGroundLevel : " << m_ori_level;
  }

  for (Integer i = m_min_level; i <= m_max_level; ++i) {
    tm->info() << "Level " << i << " : ";
    tm->info() << "\tUID Cells : [" << firstCellUniqueId(i) << ", " << (firstCellUniqueId(i) + nbCellInLevel(i)) << "[";
    tm->info() << "\tUID Nodes : [" << firstNodeUniqueId(i) << ", " << (firstNodeUniqueId(i) + nbNodeInLevel(i)) << "[";
    tm->info() << "\tUID Faces : [" << firstFaceUniqueId(i) << ", " << (firstFaceUniqueId(i) + nbFaceInLevel(i)) << "[";
  }

  const auto* m_generation_info = ICartesianMeshGenerationInfo::getReference(m_mesh, true);

  Int64ConstArrayView global_nb_cells_by_direction = m_generation_info->globalNbCells();
  tm->info() << "global_nb_cells_by_direction.x : " << global_nb_cells_by_direction[MD_DirX];
  tm->info() << "global_nb_cells_by_direction.y : " << global_nb_cells_by_direction[MD_DirY];
  tm->info() << "global_nb_cells_by_direction.z : " << global_nb_cells_by_direction[MD_DirZ];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
prepareLevel(Int32 level)
{
  if (level <= m_max_level && level >= m_min_level)
    return;
  if (level == m_max_level + 1) {
    m_max_level++;
    m_first_cell_uid_level.add(m_latest_cell_uid);
    m_first_node_uid_level.add(m_latest_node_uid);
    m_first_face_uid_level.add(m_latest_face_uid);
  }
  else if (level == m_min_level - 1) {
    m_min_level--;
    _pushFront(m_first_cell_uid_level, m_latest_cell_uid);
    _pushFront(m_first_node_uid_level, m_latest_node_uid);
    _pushFront(m_first_face_uid_level, m_latest_face_uid);
  }
  else {
    ARCANE_FATAL("Level error : {0}", level);
  }

  m_latest_cell_uid += nbCellInLevel(level);
  m_latest_node_uid += nbNodeInLevel(level);
  m_latest_face_uid += nbFaceInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
updateFirstLevel()
{
  const Int32 nb_levels_to_add = -m_min_level;
  m_ori_level += nb_levels_to_add;

  if (nb_levels_to_add == 0) {
    return;
  }

  m_max_level += nb_levels_to_add;
  m_min_level += nb_levels_to_add;

  const Integer to_div = m_pattern * nb_levels_to_add;
  if (m_dimension == 2) {
    m_nb_cell_ground.x /= to_div;
    m_nb_cell_ground.y /= to_div;
    // z reste à 1.
  }
  else {
    m_nb_cell_ground /= to_div;
  }

  // ----------
  // CartesianMeshCoarsening2::_recomputeMeshGenerationInfo()
  // Recalcule les informations sur le nombre de mailles par direction.
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(m_mesh, false);
  if (!cmgi)
    return;

  {
    ConstArrayView<Int64> v = cmgi->ownCellOffsets();
    cmgi->setOwnCellOffsets(v[0] / to_div, v[1] / to_div, v[2] / to_div);
  }
  {
    ConstArrayView<Int64> v = cmgi->globalNbCells();
    cmgi->setGlobalNbCells(v[0] / to_div, v[1] / to_div, v[2] / to_div);
  }
  {
    ConstArrayView<Int32> v = cmgi->ownNbCells();
    cmgi->setOwnNbCells(v[0] / to_div, v[1] / to_div, v[2] / to_div);
  }
  cmgi->setFirstOwnCellUniqueId(firstCellUniqueId(0));
  // CartesianMeshCoarsening2::_recomputeMeshGenerationInfo()
  // ----------
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
firstCellUniqueId(Int32 level) const
{
  return m_first_cell_uid_level[level - m_min_level];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
firstNodeUniqueId(Int32 level) const
{
  return m_first_node_uid_level[level - m_min_level];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
firstFaceUniqueId(Int32 level) const
{
  return m_first_face_uid_level[level - m_min_level];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbCellsX(Int32 level) const
{
  return static_cast<CartCoord>(static_cast<Real>(m_nb_cell_ground.x) * std::pow(m_pattern, level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbCellsY(Int32 level) const
{
  return static_cast<CartCoord>(static_cast<Real>(m_nb_cell_ground.y) * std::pow(m_pattern, level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbCellsZ(Int32 level) const
{
  return static_cast<CartCoord>(static_cast<Real>(m_nb_cell_ground.z) * std::pow(m_pattern, level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbNodesX(Int32 level) const
{
  return globalNbCellsX(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbNodesY(Int32 level) const
{
  return globalNbCellsY(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbNodesZ(Int32 level) const
{
  return globalNbCellsZ(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesX(Int32 level) const
{
  return globalNbCellsX(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesY(Int32 level) const
{
  return globalNbCellsY(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesZ(Int32 level) const
{
  return globalNbCellsZ(level) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesXCartesianView(Int32 level) const
{
  return (globalNbCellsX(level) * 2) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesYCartesianView(Int32 level) const
{
  return (globalNbCellsY(level) * 2) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
globalNbFacesZCartesianView(Int32 level) const
{
  return (globalNbCellsZ(level) * 2) + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
nbCellInLevel(Int32 level) const
{
  if (m_dimension == 2) {
    const Int64x2 nb_cell(globalNbCellsX(level), globalNbCellsY(level));
    return nb_cell.x * nb_cell.y;
  }

  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));
  return nb_cell.x * nb_cell.y * nb_cell.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
nbNodeInLevel(Int32 level) const
{
  if (m_dimension == 2) {
    const Int64x2 nb_cell(globalNbCellsX(level), globalNbCellsY(level));
    return (nb_cell.x + 1) * (nb_cell.y + 1);
  }

  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));
  return (nb_cell.x + 1) * (nb_cell.y + 1) * (nb_cell.z + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
nbFaceInLevel(Int32 level) const
{
  if (m_dimension == 2) {
    const Int64x2 nb_cell(globalNbCellsX(level), globalNbCellsY(level));
    return (nb_cell.x * nb_cell.y) * 2 + nb_cell.x * 2 + nb_cell.y;
  }

  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));
  return (nb_cell.x + 1) * nb_cell.y * nb_cell.z + (nb_cell.y + 1) * nb_cell.z * nb_cell.x + (nb_cell.z + 1) * nb_cell.x * nb_cell.y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
pattern() const
{
  return m_pattern;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
cellLevel(Int64 uid) const
{
  Int32 pos = -1;
  Int64 max = -1;

  for (Int32 i = m_min_level; i <= m_max_level; ++i) {
    const Int64 first_uid = firstCellUniqueId(i);
    if (first_uid <= uid && first_uid > max) {
      pos = i;
      max = first_uid;
    }
  }
#ifdef ARCANE_CHECK
  if (max == -1) {
    ARCANE_FATAL("CellUID is not in any patch (UID too low)");
  }
  if (uid >= firstCellUniqueId(pos) + nbCellInLevel(pos)) {
    ARCANE_FATAL("CellUID is not in any patch (UID too high)");
  }
#endif
  return pos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
nodeLevel(Int64 uid) const
{
  Int32 pos = -1;
  Int64 max = -1;

  for (Int32 i = m_min_level; i <= m_max_level; ++i) {
    const Int64 first_uid = firstNodeUniqueId(i);
    if (first_uid <= uid && first_uid > max) {
      pos = i;
      max = first_uid;
    }
  }
#ifdef ARCANE_CHECK
  if (max == -1) {
    ARCANE_FATAL("NodeUID is not in any patch (UID too low)");
  }
  if (uid >= firstNodeUniqueId(pos) + nbNodeInLevel(pos)) {
    ARCANE_FATAL("NodeUID is not in any patch (UID too high)");
  }
#endif
  return pos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
faceLevel(Int64 uid) const
{
  Int32 pos = -1;
  Int64 max = -1;

  for (Int32 i = m_min_level; i <= m_max_level; ++i) {
    const Int64 first_uid = firstFaceUniqueId(i);
    if (first_uid <= uid && first_uid > max) {
      pos = i;
      max = first_uid;
    }
  }
#ifdef ARCANE_CHECK
  if (max == -1) {
    ARCANE_FATAL("FaceUID is not in any patch (UID too low)");
  }
  if (uid >= firstFaceUniqueId(pos) + nbFaceInLevel(pos)) {
    ARCANE_FATAL("FaceUID is not in any patch (UID too high)");
  }
#endif
  return pos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Tant que l'on a un unique "pattern" pour x, y, z, pas besoin de trois méthodes.
CartCoord CartesianMeshNumberingMngInternal::
offsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const
{
  if (level_from == level_to) {
    return coord;
  }
  if (level_from < level_to) {
    return coord * static_cast<CartCoord>(std::pow(m_pattern, (level_to - level_from)));
  }
  return coord / static_cast<CartCoord>(std::pow(m_pattern, (level_from - level_to)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3 CartesianMeshNumberingMngInternal::
offsetLevelToLevel(CartCoord3 coord, Int32 level_from, Int32 level_to) const
{
  if (level_from == level_to) {
    return coord;
  }
  if (level_from < level_to) {
    return coord * static_cast<CartCoord>(std::pow(m_pattern, (level_to - level_from)));
  }
  return coord / static_cast<CartCoord>(std::pow(m_pattern, (level_from - level_to)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Tant que l'on a un unique "pattern" pour x, y, z, pas besoin de trois méthodes.
CartCoord CartesianMeshNumberingMngInternal::
faceOffsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const
{
  // Admettons que l'on ai les faces suivantes :
  //  ┌─0──┬──2─┐
  // 4│   6│   8│
  //  ├─5──┼─7──┤
  // 9│  11│  13│
  //  └─10─┴─12─┘

  // Pour la position d'une face, on considère cette disposition :
  // ┌──┬──┬──┬──┬──┐
  // │  │ 0│  │ 2│  │
  // ├──┼──┼──┼──┼──┤
  // │ 4│  │ 6│  │ 8│
  // ├──┼──┼──┼──┼──┤
  // │  │ 5│  │ 7│  │
  // ├──┼──┼──┼──┼──┤
  // │ 9│  │11│  │13│
  // ├──┼──┼──┼──┼──┤
  // │  │10│  │12│  │
  // └──┴──┴──┴──┴──┘

  if (level_from == level_to) {
    return coord;
  }

  if (level_from < level_to) {
    const Int32 pattern = m_pattern * (level_to - level_from);
    if (coord % 2 == 0) {
      return coord * pattern;
    }
    return ((coord - 1) * pattern) + 1;
  }

  const Int32 pattern = m_pattern * (level_from - level_to);
  if (coord % 2 == 0) {
    if (coord % (pattern * 2) == 0) {
      return coord / pattern;
    }
    return -1;
  }

  //    auto a = coord - 1;
  //    auto b = a % (pattern * 2);
  //    auto c = a / (pattern * 2);
  //    auto d = c * (2 * (pattern - 1));
  //    auto e = d + b;
  //    auto f = coord - e;
  //    return f;

  return coord - (((coord - 1) / (pattern * 2) * (2 * (pattern - 1))) + ((coord - 1) % (pattern * 2)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3 CartesianMeshNumberingMngInternal::
cellUniqueIdToCoord(Int64 uid, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);

  return { static_cast<CartCoord>(to2d % nb_cell_x), static_cast<CartCoord>(to2d / nb_cell_x), static_cast<CartCoord>(uid / (nb_cell_x * nb_cell_y)) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3 CartesianMeshNumberingMngInternal::
cellUniqueIdToCoord(Cell cell)
{
  return cellUniqueIdToCoord(cell.uniqueId().asInt64(), cell.level());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordX(Int64 uid, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return static_cast<CartCoord>(to2d % nb_cell_x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordX(Cell cell)
{
  return cellUniqueIdToCoordX(cell.uniqueId(), cell.level());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordY(Int64 uid, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  const Int64 to2d = uid % (nb_cell_x * nb_cell_y);
  return static_cast<CartCoord>(to2d / nb_cell_x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordY(Cell cell)
{
  return cellUniqueIdToCoordY(cell.uniqueId(), cell.level());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordZ(Int64 uid, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  uid -= first_cell_uid;

  return static_cast<CartCoord>(uid / (nb_cell_x * nb_cell_y));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
cellUniqueIdToCoordZ(Cell cell)
{
  return cellUniqueIdToCoordZ(cell.uniqueId(), cell.level());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordX(Int64 uid, Int32 level)
{
  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 nb_node_y = globalNbNodesY(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  uid -= first_node_uid;

  const Int64 to2d = uid % (nb_node_x * nb_node_y);
  return static_cast<CartCoord>(to2d % nb_node_x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordX(Node node)
{
  const Int64 uid = node.uniqueId().asInt64();
  return nodeUniqueIdToCoordX(uid, nodeLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordY(Int64 uid, Int32 level)
{
  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 nb_node_y = globalNbNodesY(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  uid -= first_node_uid;

  const Int64 to2d = uid % (nb_node_x * nb_node_y);
  return static_cast<CartCoord>(to2d / nb_node_x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordY(Node node)
{
  const Int64 uid = node.uniqueId().asInt64();
  return nodeUniqueIdToCoordY(uid, nodeLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordZ(Int64 uid, Int32 level)
{
  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 nb_node_y = globalNbNodesY(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  uid -= first_node_uid;

  return static_cast<CartCoord>(uid / (nb_node_x * nb_node_y));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
nodeUniqueIdToCoordZ(Node node)
{
  const Int64 uid = node.uniqueId().asInt64();
  return nodeUniqueIdToCoordZ(uid, nodeLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordX(Int64 uid, Int32 level)
{
  if (m_dimension == 2) {

    const Int64 nb_face_x = globalNbFacesXCartesianView(level);
    const Int64 first_face_uid = firstFaceUniqueId(level);

    uid -= first_face_uid;
    uid += 1;

    // Le +1 nous permet d'avoir le niveau (imaginaire) -1 commençant à 0 :
    //
    //     x =  0  1  2  3  4
    //        ┌──┬──┬──┬──┬──┐
    // y = -1 │ 0│  │ 2│  │ 4│
    //        ┌──┬──┬──┬──┬──┐
    // y = 0  │  │ 1│  │ 3│  │
    //        ├──┼──┼──┼──┼──┤
    // y = 1  │ 5│  │ 7│  │ 9│
    //        ├──┼──┼──┼──┼──┤
    // y = 2  │  │ 6│  │ 8│  │
    //        ├──┼──┼──┼──┼──┤
    // y = 3  │10│  │12│  │14│
    //        ├──┼──┼──┼──┼──┤
    // y = 4  │  │11│  │13│  │
    //        └──┴──┴──┴──┴──┘
    //
    // Si on fait "tomber" les faces (tetris), on obtient une numérotation
    // cartésienne classique.

    return static_cast<CartCoord>(uid % nb_face_x);
  }

  const Int64 nb_face_x = globalNbFacesX(level);
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 first_face_uid = firstFaceUniqueId(level);

  //    Int64 initial_uid = uid;

  uid -= first_face_uid;

  Int64x3 three_parts_numbering = _face3DNumberingThreeParts(level);

  // Prenons la vue des faces en grille cartésienne d'un maillage 2x2x2 :
  //         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
  //      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
  //         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
  //  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
  //         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
  //                          │                  │                  │                  │
  //
  // On peut remarquer 3 "sortes de disposition" : les faces ayant les uid [0, 11],
  // les faces ayant les uid [12, 23] et les faces ayant les uid [24, 35].
  // On récupère ces intervalles avec la méthode face3DNumberingThreeParts().

  // Pour l'intervalle [0, 11], on remarque que l'origine en X est toujours 0 et que les mailles
  // contenant une face sont sur les X pairs.
  // Enfin, on a "nb_face_x" faces en X.
  if (uid < three_parts_numbering.x) {

    //      debug() << "faceUniqueIdToCoordX (2)"
    //              << " -- true uid : " << initial_uid
    //              << " -- uid : " << uid
    //              << " -- level : " << level
    //              << " -- three_parts_numbering : " << three_parts_numbering
    //              << " -- nb_face_x : " << nb_face_x
    //              << " -- return : " << ((uid % nb_face_x) * 2);

    return static_cast<CartCoord>((uid % nb_face_x) * 2);
  }

  // Pour l'intervalle [12, 23], on remarque que l'origine en X est toujours 1 et que les mailles
  // contenant une face sont sur les X impairs.
  // Enfin, on a "nb_cell_x" faces en X.
  if (uid < three_parts_numbering.x + three_parts_numbering.y) {
    uid -= three_parts_numbering.x;
    //      debug() << "faceUniqueIdToCoordX (3)"
    //              << " -- true uid : " << initial_uid
    //              << " -- uid : " << uid
    //              << " -- level : " << level
    //              << " -- three_parts_numbering : " << three_parts_numbering
    //              << " -- nb_cell_x : " << nb_cell_x
    //              << " -- return : " << ((uid % nb_cell_x) * 2 + 1);

    return static_cast<CartCoord>((uid % nb_cell_x) * 2 + 1);
  }

  // Pour l'intervalle [24, 35], on remarque que l'origine en X est toujours 1 et que les mailles
  // contenant une face sont sur les X impairs.
  // Enfin, on a "nb_cell_x" faces en X.
  uid -= three_parts_numbering.x + three_parts_numbering.y;
  //      debug() << "faceUniqueIdToCoordX (1)"
  //              << " -- true uid : " << initial_uid
  //              << " -- uid : " << uid
  //              << " -- level : " << level
  //              << " -- three_parts_numbering : " << three_parts_numbering
  //              << " -- nb_cell_x : " << nb_cell_x
  //              << " -- return : " << ((uid % nb_cell_x) * 2 + 1);

  return static_cast<CartCoord>((uid % nb_cell_x) * 2 + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordX(Face face)
{
  const Int64 uid = face.uniqueId().asInt64();
  return faceUniqueIdToCoordX(uid, faceLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordY(Int64 uid, Int32 level)
{
  if (m_dimension == 2) {
    const Int64 nb_face_x = globalNbFacesXCartesianView(level);
    const Int64 first_face_uid = firstFaceUniqueId(level);

    uid -= first_face_uid;
    uid += 1;

    // Le +1 nous permet d'avoir le niveau (imaginaire) -1 commençant à 0 :
    //
    //     x =  0  1  2  3  4
    //        ┌──┬──┬──┬──┬──┐
    // y = -1 │ 0│  │ 2│  │ 4│
    //        ┌──┬──┬──┬──┬──┐
    // y = 0  │  │ 1│  │ 3│  │
    //        ├──┼──┼──┼──┼──┤
    // y = 1  │ 5│  │ 7│  │ 9│
    //        ├──┼──┼──┼──┼──┤
    // y = 2  │  │ 6│  │ 8│  │
    //        ├──┼──┼──┼──┼──┤
    // y = 3  │10│  │12│  │14│
    //        ├──┼──┼──┼──┼──┤
    // y = 4  │  │11│  │13│  │
    //        └──┴──┴──┴──┴──┘
    //
    // Si, en plus, on fait y+1, on simplifie le problème puisque si on fait
    // "tomber" les faces (tetris), on obtient une numérotation cartesienne classique.

    const Int64 flat_pos = uid / nb_face_x;
    return static_cast<CartCoord>((flat_pos * 2) + (flat_pos % 2 == uid % 2 ? 0 : 1) - 1); // Le -1 pour "retirer" le niveau imaginaire.
  }
  const Int64 nb_face_x = globalNbFacesX(level);
  const Int64 nb_face_y = globalNbFacesY(level);
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_face_uid = firstFaceUniqueId(level);

  //    Int64 initial_uid = uid;

  uid -= first_face_uid;

  Int64x3 three_parts_numbering = _face3DNumberingThreeParts(level);

  // Prenons la vue des faces en grille cartésienne d'un maillage 2x2x2 :
  //         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
  //      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
  //         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
  //  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
  //         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
  //                          │                  │                  │                  │
  //
  // On peut remarquer 3 "sortes de disposition" : les faces ayant les uid [0, 11],
  // les faces ayant les uid [12, 23] et les faces ayant les uid [24, 35].
  // On récupère ces intervalles avec la méthode face3DNumberingThreeParts().

  // Pour l'intervalle [0, 1], on remarque que l'origine en Y est toujours 1 et que les mailles
  // contenant une face sont sur les Y impairs.
  // Enfin, on a "nb_cell_y" faces en Y.
  if (uid < three_parts_numbering.x) {
    uid %= nb_face_x * nb_cell_y;

    //      debug() << "faceUniqueIdToCoordY (2)"
    //              << " -- true uid : " << initial_uid
    //              << " -- uid : " << uid
    //              << " -- level : " << level
    //              << " -- three_parts_numbering : " << three_parts_numbering
    //              << " -- nb_face_x : " << nb_face_x
    //              << " -- nb_cell_y : " << nb_cell_y
    //              << " -- return : " << ((uid / nb_face_x) * 2 + 1);

    return static_cast<CartCoord>((uid / nb_face_x) * 2 + 1);
  }

  // Pour l'intervalle [12, 23], on remarque que l'origine en Y est toujours 0 et que les mailles
  // contenant une face sont sur les Y pairs.
  // Enfin, on a "nb_face_y" faces en Y.
  if (uid < three_parts_numbering.x + three_parts_numbering.y) {
    uid -= three_parts_numbering.x;
    uid %= nb_cell_x * nb_face_y;

    //      debug() << "faceUniqueIdToCoordY (3)"
    //              << " -- true uid : " << initial_uid
    //              << " -- uid : " << uid
    //              << " -- level : " << level
    //              << " -- three_parts_numbering : " << three_parts_numbering
    //              << " -- nb_cell_x : " << nb_cell_x
    //              << " -- nb_face_y : " << nb_face_y
    //              << " -- return : " << ((uid / nb_cell_x) * 2);

    return static_cast<CartCoord>((uid / nb_cell_x) * 2);
  }

  // Pour l'intervalle [24, 35], on remarque que l'origine en Y est toujours 1 et que les mailles
  // contenant une face sont sur les Y impairs.
  // Enfin, on a "nb_cell_y" faces en Y.
  uid -= three_parts_numbering.x + three_parts_numbering.y;
  uid %= nb_cell_x * nb_cell_y;

  //      debug() << "faceUniqueIdToCoordY (1)"
  //              << " -- true uid : " << initial_uid
  //              << " -- uid : " << uid
  //              << " -- level : " << level
  //              << " -- three_parts_numbering : " << three_parts_numbering
  //              << " -- nb_cell_x : " << nb_cell_x
  //              << " -- nb_cell_y : " << nb_cell_y
  //              << " -- return : " << ((uid / nb_cell_x) * 2 + 1);

  return static_cast<CartCoord>((uid / nb_cell_x) * 2 + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordY(Face face)
{
  const Int64 uid = face.uniqueId().asInt64();
  return faceUniqueIdToCoordY(uid, faceLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordZ(Int64 uid, Int32 level)
{
  const Int64 nb_face_x = globalNbFacesX(level);
  const Int64 nb_face_y = globalNbFacesY(level);
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_face_uid = firstFaceUniqueId(level);

  //  Int64 initial_uid = uid;

  uid -= first_face_uid;

  Int64x3 three_parts_numbering = _face3DNumberingThreeParts(level);

  // Prenons la vue des faces en grille cartésienne d'un maillage 2x2x2 :
  //         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
  //      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
  //         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
  //  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
  //         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
  //                          │                  │                  │                  │
  //
  // On peut remarquer 3 "sortes de disposition" : les faces ayant les uid [0, 11],
  // les faces ayant les uid [12, 23] et les faces ayant les uid [24, 35].
  // On récupère ces intervalles avec la méthode face3DNumberingThreeParts().

  // Pour l'intervalle [0, 11], on remarque que l'origine en Z est toujours 1 et que les mailles
  // contenant une face sont sur les Z impairs.
  // Enfin, on a "nb_cell_z" faces en Z.
  if (uid < three_parts_numbering.x) {

    //    debug() << "faceUniqueIdToCoordZ (2)"
    //            << " -- true uid : " << initial_uid
    //            << " -- uid : " << uid
    //            << " -- level : " << level
    //            << " -- three_parts_numbering : " << three_parts_numbering
    //            << " -- nb_face_x : " << nb_face_x
    //            << " -- nb_cell_y : " << nb_cell_y
    //            << " -- return : " << ((uid / (nb_face_x * nb_cell_y)) * 2 + 1);

    return static_cast<CartCoord>((uid / (nb_face_x * nb_cell_y)) * 2 + 1);
  }

  // Pour l'intervalle [12, 23], on remarque que l'origine en Z est toujours 1 et que les mailles
  // contenant une face sont sur les Z impairs.
  // Enfin, on a "nb_cell_z" faces en Z.
  if (uid < three_parts_numbering.x + three_parts_numbering.y) {
    uid -= three_parts_numbering.x;

    //    debug() << "faceUniqueIdToCoordZ (3)"
    //            << " -- true uid : " << initial_uid
    //            << " -- uid : " << uid
    //            << " -- level : " << level
    //            << " -- three_parts_numbering : " << three_parts_numbering
    //            << " -- nb_cell_x : " << nb_cell_x
    //            << " -- nb_face_y : " << nb_face_y
    //            << " -- return : " << ((uid / (nb_cell_x * nb_face_y)) * 2 + 1);

    return static_cast<CartCoord>((uid / (nb_cell_x * nb_face_y)) * 2 + 1);
  }

  // Pour l'intervalle [24, 35], on remarque que l'origine en Z est toujours 0 et que les mailles
  // contenant une face sont sur les Z pairs.
  // Enfin, on a "nb_face_z" faces en Z.
  uid -= three_parts_numbering.x + three_parts_numbering.y;

  //    debug() << "faceUniqueIdToCoordZ (1)"
  //            << " -- true uid : " << initial_uid
  //            << " -- uid : " << uid
  //            << " -- level : " << level
  //            << " -- three_parts_numbering : " << three_parts_numbering
  //            << " -- nb_cell_x : " << nb_cell_x
  //            << " -- nb_cell_y : " << nb_cell_y
  //            << " -- return : " << ((uid / (nb_cell_x * nb_cell_y)) * 2);

  return static_cast<CartCoord>((uid / (nb_cell_x * nb_cell_y)) * 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMngInternal::
faceUniqueIdToCoordZ(Face face)
{
  const Int64 uid = face.uniqueId().asInt64();
  return faceUniqueIdToCoordZ(uid, faceLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
cellUniqueId(CartCoord3 cell_coord, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  return (cell_coord.x + cell_coord.y * nb_cell_x + cell_coord.z * nb_cell_x * nb_cell_y) + first_cell_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
cellUniqueId(CartCoord2 cell_coord, Int32 level)
{
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 first_cell_uid = firstCellUniqueId(level);

  return (cell_coord.x + cell_coord.y * nb_cell_x) + first_cell_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
nodeUniqueId(CartCoord3 node_coord, Int32 level)
{
  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 nb_node_y = globalNbNodesY(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  return (node_coord.x + node_coord.y * nb_node_x + node_coord.z * nb_node_x * nb_node_y) + first_node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
nodeUniqueId(CartCoord2 node_coord, Int32 level)
{
  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  return (node_coord.x + node_coord.y * nb_node_x) + first_node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
faceUniqueId(CartCoord3 face_coord, Int32 level)
{
  const Int64 nb_face_x = globalNbFacesX(level);
  const Int64 nb_face_y = globalNbFacesY(level);
  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_cell_y = globalNbCellsY(level);

  Int64x3 three_parts_numbering = _face3DNumberingThreeParts(level);
  Int64 uid = firstFaceUniqueId(level);

  // Prenons la vue des faces en grille cartésienne d'un maillage 2x2x2 :
  //         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
  //      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
  //         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
  //  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
  //         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
  //                          │                  │                  │                  │
  //
  // On peut remarquer 3 "sortes de disposition" : les faces ayant les uid [0, 11],
  // les faces ayant les uid [12, 23] et les faces ayant les uid [24, 35].
  // Ici, on veut faire l'inverse : passer des coordonnées x,y,z à un uid.
  // Pour identifier les trois intervalles, on regarde quelle coordonnée est pair.
  // En effet, pour l'intervalle [0, 11], on s'aperçoit que seule la coordonnée z est pair.
  // Pour l'intervalle [12, 23], seule la coordonnée x est pair et pour l'intervalle [24, 35],
  // seule la coordonnée y est pair.

  // Intervalle [0, 11].
  if (face_coord.x % 2 == 0) {

    // Ici, on place les mailles à l'origine 0*0*0 et on les met côte-à-côte.
    face_coord.y -= 1;
    face_coord.z -= 1;

    face_coord /= 2;

    // On est, à présent et pour cet intervalle, dans une vue cartésienne de
    // taille nb_face_x * nb_cell_y * nb_cell_z.
    uid += face_coord.x + (face_coord.y * nb_face_x) + (face_coord.z * nb_face_x * nb_cell_y);
  }

  // Intervalle [12, 23].
  else if (face_coord.y % 2 == 0) {
    uid += three_parts_numbering.x;

    // Ici, on place les mailles à l'origine 0*0*0 et on les met côte-à-côte.
    face_coord.x -= 1;
    face_coord.z -= 1;

    face_coord /= 2;

    // On est, à présent et pour cet intervalle, dans une vue cartésienne de
    // taille nb_cell_x * nb_face_y * nb_cell_z.
    uid += face_coord.x + (face_coord.y * nb_cell_x) + (face_coord.z * nb_cell_x * nb_face_y);
  }

  // Intervalle [24, 35].
  else if (face_coord.z % 2 == 0) {
    uid += three_parts_numbering.x + three_parts_numbering.y;

    // Ici, on place les mailles à l'origine 0*0*0 et on les met côte-à-côte.
    face_coord.x -= 1;
    face_coord.y -= 1;

    face_coord /= 2;

    // On est, à présent et pour cet intervalle, dans une vue cartésienne de
    // taille nb_cell_x * nb_cell_y * nb_face_z.
    uid += face_coord.x + (face_coord.y * nb_cell_x) + (face_coord.z * nb_cell_x * nb_cell_y);
  }
  else {
    ARCANE_FATAL("Bizarre -- x : {0} -- y : {1} -- z : {2}", face_coord.x, face_coord.y, face_coord.z);
  }

  return uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
faceUniqueId(CartCoord2 face_coord, Int32 level)
{
  const Int64 nb_face_x = globalNbFacesXCartesianView(level);
  const Int64 first_face_uid = firstFaceUniqueId(level);

  // On considère que l'on a un niveau imaginaire -1 et que
  // l'on obtiendra uid+1 (dans la numérotation utilisée normalement,
  // la face à la position (1, 0) a un uid = 0).
  //
  //     x =  0  1  2  3  4
  //        ┌──┬──┬──┬──┬──┐
  // y = -1 │ 0│  │ 2│  │ 4│
  //        ┌──┬──┬──┬──┬──┐
  // y = 0  │  │ 1│  │ 3│  │
  //        ├──┼──┼──┼──┼──┤
  // y = 1  │ 5│  │ 7│  │ 9│
  //        ├──┼──┼──┼──┼──┤
  // y = 2  │  │ 6│  │ 8│  │
  //        ├──┼──┼──┼──┼──┤
  // y = 3  │10│  │12│  │14│
  //        ├──┼──┼──┼──┼──┤
  // y = 4  │  │11│  │13│  │
  //        └──┴──┴──┴──┴──┘
  //

  face_coord.y += 1;

  const Int64 a = (face_coord.y / 2) * nb_face_x;

  return (face_coord.x + a - 1) + first_face_uid; // Le -1 est pour revenir à la numérotation normale.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
nbNodeByCell()
{
  return (m_dimension == 2 ? 4 : 8);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellNodeUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid)
{
  if (uid.size() != nbNodeByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 nb_node_y = globalNbNodesY(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  const Int64 x0 = cell_coord.x + 0;
  const Int64 x1 = cell_coord.x + 1;

  const Int64 y0 = (cell_coord.y + 0) * nb_node_x;
  const Int64 y1 = (cell_coord.y + 1) * nb_node_x;

  const Int64 z0 = (cell_coord.z + 0) * nb_node_x * nb_node_y;
  const Int64 z1 = (cell_coord.z + 1) * nb_node_x * nb_node_y;

  uid[0] = x0 + y0 + z0 + first_node_uid;
  uid[1] = x1 + y0 + z0 + first_node_uid;
  uid[2] = x1 + y1 + z0 + first_node_uid;
  uid[3] = x0 + y1 + z0 + first_node_uid;

  uid[4] = x0 + y0 + z1 + first_node_uid;
  uid[5] = x1 + y0 + z1 + first_node_uid;
  uid[6] = x1 + y1 + z1 + first_node_uid;
  uid[7] = x0 + y1 + z1 + first_node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellNodeUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid)
{
  if (uid.size() != nbNodeByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_node_x = globalNbNodesX(level);
  const Int64 first_node_uid = firstNodeUniqueId(level);

  const Int64 x0 = cell_coord.x + 0;
  const Int64 x1 = cell_coord.x + 1;

  const Int64 y0 = (cell_coord.y + 0) * nb_node_x;
  const Int64 y1 = (cell_coord.y + 1) * nb_node_x;

  uid[0] = x0 + y0 + first_node_uid;
  uid[1] = x1 + y0 + first_node_uid;
  uid[2] = x1 + y1 + first_node_uid;
  uid[3] = x0 + y1 + first_node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid)
{
  if (m_dimension == 2) {
    const CartCoord2 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level));
    cellNodeUniqueIds(cell_coord, level, uid);
  }
  else {
    const CartCoord3 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level), cellUniqueIdToCoordZ(cell_uid, level));
    cellNodeUniqueIds(cell_coord, level, uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid)
{
  cellNodeUniqueIds(cell.uniqueId().asInt64(), cell.level(), uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMngInternal::
nbFaceByCell()
{
  return (m_dimension == 2 ? 4 : 6);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellFaceUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid)
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
  // même (en notation localId Arcane (cell.face(i)) : 1, 4, 2, 5, 0, 3).
  // Les UniqueIDs générés sont donc les mêmes quelque soit le découpage.
  //
  // Prenons la vue des faces en grille cartésienne d'un maillage 2x2x2 :
  //         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
  //      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
  //         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
  //  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
  //         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
  //  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
  //         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
  //                          │                  │                  │                  │
  //
  // (dans cette vue, les mailles se situent aux X, Y et Z impaires
  // (donc ici, [1, 1, 1], [3, 1, 1], [1, 3, 1], &c)).
  //
  // On a un cube décomposé en huit cellules (2x2x2).
  // Le schéma au-dessus représente les faces des cellules de ce cube avec
  // les uniqueIDs que l'algorithme génèrera (sans face_adder).
  // Pour cet algo, on commence par les faces "yz" (direction X).
  // On énumère d'abord en x, puis en y, puis en z.
  // Une fois les faces "yz" numérotées, on fait les faces "zx" (direction Y).
  // Toujours le même ordre de numérotation.
  // On termine avec les faces "xy" (direction Z), encore dans le même ordre.
  //
  // Dans l'implémentation ci-dessous, on fait la numérotation
  // maille par maille.

  const Int64 total_face_yz = nb_face.x * nb_cell.y * nb_cell.z;
  const Int64 total_face_yz_zx = total_face_yz + nb_face.y * nb_cell.z * nb_cell.x;

  const Int64 nb_cell_before_j = cell_coord.y * nb_cell.x;

  uid[0] = (cell_coord.z * nb_cell.x * nb_cell.y) + nb_cell_before_j + cell_coord.x + total_face_yz_zx;

  uid[3] = uid[0] + nb_cell.x * nb_cell.y;

  uid[1] = (cell_coord.z * nb_face.x * nb_cell.y) + (cell_coord.y * nb_face.x) + cell_coord.x;

  uid[4] = uid[1] + 1;

  uid[2] = (cell_coord.z * nb_cell.x * nb_face.y) + nb_cell_before_j + cell_coord.x + total_face_yz;

  uid[5] = uid[2] + nb_cell.x;

  uid[0] += first_face_uid;
  uid[1] += first_face_uid;
  uid[2] += first_face_uid;
  uid[3] += first_face_uid;
  uid[4] += first_face_uid;
  uid[5] += first_face_uid;

  // debug() << "Coord : " << cell_coord << " -- UIDs : " << uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellFaceUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid)
{
  if (uid.size() != nbFaceByCell())
    ARCANE_FATAL("Bad size of arrayview");

  const Int64 nb_cell_x = globalNbCellsX(level);
  const Int64 nb_face_x = nb_cell_x + 1;
  const Int64 first_face_uid = firstFaceUniqueId(level);

  // Numérote les faces
  //  ┌─0──┬──2─┐
  // 4│   6│   8│
  //  ├─5──┼─7──┤
  // 9│  11│  13│
  //  └─10─┴─12─┘
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid)
{
  if (m_dimension == 2) {
    const CartCoord2 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level));
    cellFaceUniqueIds(cell_coord, level, uid);
  }
  else {
    const CartCoord3 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level), cellUniqueIdToCoordZ(cell_uid, level));
    cellFaceUniqueIds(cell_coord, level, uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid)
{
  cellFaceUniqueIds(cell.uniqueId().asInt64(), cell.level(), uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundCell(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid)
{
  ARCANE_ASSERT((uid.size() == 27), ("Size of uid array != 27"));

  uid.fill(-1);

  const CartCoord nb_cells_x = globalNbCellsX(level);
  const CartCoord nb_cells_y = globalNbCellsY(level);
  const CartCoord nb_cells_z = globalNbCellsZ(level);

  for (CartCoord k = -1; k < 2; ++k) {
    const CartCoord coord_around_cell_z = cell_coord.z + k;
    if (coord_around_cell_z >= 0 && coord_around_cell_z < nb_cells_z) {
      for (CartCoord j = -1; j < 2; ++j) {
        const CartCoord coord_around_cell_y = cell_coord.y + j;

        if (coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y) {
          for (CartCoord i = -1; i < 2; ++i) {
            const CartCoord coord_around_cell_x = cell_coord.x + i;

            if (coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
              uid[(i + 1) + ((j + 1) * 3) + ((k + 1) * 9)] = cellUniqueId(CartCoord3{ coord_around_cell_x, coord_around_cell_y, coord_around_cell_z }, level);
            }
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundCell(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid)
{
  ARCANE_ASSERT((uid.size() == 9), ("Size of uid array != 9"));

  uid.fill(-1);

  const CartCoord nb_cells_x = globalNbCellsX(level);
  const CartCoord nb_cells_y = globalNbCellsY(level);

  for (CartCoord j = -1; j < 2; ++j) {
    const CartCoord coord_around_cell_y = cell_coord.y + j;
    if (coord_around_cell_y >= 0 && coord_around_cell_y < nb_cells_y) {

      for (CartCoord i = -1; i < 2; ++i) {
        const CartCoord coord_around_cell_x = cell_coord.x + i;
        if (coord_around_cell_x >= 0 && coord_around_cell_x < nb_cells_x) {
          uid[(i + 1) + ((j + 1) * 3)] = cellUniqueId(CartCoord2{ coord_around_cell_x, coord_around_cell_y }, level);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid)
{
  if (m_dimension == 2) {
    const CartCoord2 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level));
    cellUniqueIdsAroundCell(cell_coord, level, uid);
  }
  else {
    const CartCoord3 cell_coord(cellUniqueIdToCoordX(cell_uid, level), cellUniqueIdToCoordY(cell_uid, level), cellUniqueIdToCoordZ(cell_uid, level));
    cellUniqueIdsAroundCell(cell_coord, level, uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid)
{
  cellUniqueIdsAroundCell(cell.uniqueId().asInt64(), cell.level(), uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundNode(CartCoord3 node_coord, Int32 level, ArrayView<Int64> uid)
{
  ARCANE_ASSERT((uid.size() == 8), ("Size of uid array != 8"));

  uid.fill(-1);

  const CartCoord nb_cells_x = globalNbCellsX(level);
  const CartCoord nb_cells_y = globalNbCellsY(level);
  const CartCoord nb_cells_z = globalNbCellsZ(level);

  for (CartCoord k = -1; k < 1; ++k) {
    const CartCoord coord_cell_z = node_coord.z + k;
    if (coord_cell_z >= 0 && coord_cell_z < nb_cells_z) {

      for (CartCoord j = -1; j < 1; ++j) {
        const CartCoord coord_cell_y = node_coord.y + j;
        if (coord_cell_y >= 0 && coord_cell_y < nb_cells_y) {

          for (CartCoord i = -1; i < 1; ++i) {
            const CartCoord coord_cell_x = node_coord.x + i;
            if (coord_cell_x >= 0 && coord_cell_x < nb_cells_x) {
              uid[(i + 1) + ((j + 1) * 2) + ((k + 1) * 4)] = cellUniqueId(CartCoord3{ coord_cell_x, coord_cell_y, coord_cell_z }, level);
            }
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundNode(CartCoord2 node_coord, Int32 level, ArrayView<Int64> uid)
{
  ARCANE_ASSERT((uid.size() == 4), ("Size of uid array != 4"));

  uid.fill(-1);

  const CartCoord nb_cells_x = globalNbCellsX(level);
  const CartCoord nb_cells_y = globalNbCellsY(level);

  for (CartCoord j = -1; j < 1; ++j) {
    const CartCoord coord_cell_y = node_coord.y + j;
    if (coord_cell_y >= 0 && coord_cell_y < nb_cells_y) {

      for (CartCoord i = -1; i < 1; ++i) {
        const CartCoord coord_cell_x = node_coord.x + i;
        if (coord_cell_x >= 0 && coord_cell_x < nb_cells_x) {
          uid[(i + 1) + ((j + 1) * 2)] = cellUniqueId(CartCoord2{ coord_cell_x, coord_cell_y }, level);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid)
{
  if (m_dimension == 2) {
    const CartCoord2 node_coord{ nodeUniqueIdToCoordX(node_uid, level), nodeUniqueIdToCoordY(node_uid, level) };
    cellUniqueIdsAroundNode(node_coord, level, uid);
  }
  else {
    const CartCoord3 node_coord{ nodeUniqueIdToCoordX(node_uid, level), nodeUniqueIdToCoordY(node_uid, level), nodeUniqueIdToCoordZ(node_uid, level) };
    cellUniqueIdsAroundNode(node_coord, level, uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid)
{
  cellUniqueIdsAroundNode(node.uniqueId().asInt64(), nodeLevel(node.uniqueId().asInt64()), uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
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
    auto txty = [&](CartCoord pos_x, CartCoord pos_y) -> Real3 {
      const Real x = static_cast<Real>(pos_x) / static_cast<Real>(m_pattern);
      const Real y = static_cast<Real>(pos_y) / static_cast<Real>(m_pattern);

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

    constexpr CartCoord node_1d_2d_x[] = { 0, 1, 1, 0 };
    constexpr CartCoord node_1d_2d_y[] = { 0, 0, 1, 1 };

    for (CartCoord j = 0; j < m_pattern; ++j) {
      for (CartCoord i = 0; i < m_pattern; ++i) {

        Int32 begin = (i == 0 && j == 0 ? 0 : j == 0 ? 1
                                                     : 2);
        Int32 end = (i == 0 ? nbNodeByCell() : nbNodeByCell() - 1);
        Cell child = childCellOfCell(parent_cell, CartCoord2(i, j));

        for (Int32 inode = begin; inode < end; ++inode) {
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
    auto txtytz = [&](CartCoord pos_x, CartCoord pos_y, CartCoord pos_z) -> Real3 {
      const Real x = static_cast<Real>(pos_x) / static_cast<Real>(m_pattern);
      const Real y = static_cast<Real>(pos_y) / static_cast<Real>(m_pattern);
      const Real z = static_cast<Real>(pos_z) / static_cast<Real>(m_pattern);

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

    constexpr CartCoord node_1d_3d_x[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    constexpr CartCoord node_1d_3d_y[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    constexpr CartCoord node_1d_3d_z[] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    for (CartCoord k = 0; k < m_pattern; ++k) {
      for (CartCoord j = 0; j < m_pattern; ++j) {
        for (CartCoord i = 0; i < m_pattern; ++i) {

          // TODO : éviter les multiples appels pour un même noeud.
          Int32 begin = 0;
          Int32 end = nbNodeByCell();
          Cell child = childCellOfCell(parent_cell, CartCoord3(i, j, k));

          for (Int32 inode = begin; inode < end; ++inode) {
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
setParentNodeCoordinates(Cell parent_cell)
{
  if (!(parent_cell.itemBase().flags() & ItemFlags::II_JustAdded)) {
    ARCANE_FATAL("Cell not II_JustAdded");
  }

  VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();

  if (m_dimension == 2) {
    nodes_coords[parent_cell.node(0)] = nodes_coords[childCellOfCell(parent_cell, CartCoord2(0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[childCellOfCell(parent_cell, CartCoord2(m_pattern - 1, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[childCellOfCell(parent_cell, CartCoord2(m_pattern - 1, m_pattern - 1)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[childCellOfCell(parent_cell, CartCoord2(0, m_pattern - 1)).node(3)];
  }

  else {
    nodes_coords[parent_cell.node(0)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(0, 0, 0)).node(0)];
    nodes_coords[parent_cell.node(1)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(m_pattern - 1, 0, 0)).node(1)];
    nodes_coords[parent_cell.node(2)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(m_pattern - 1, m_pattern - 1, 0)).node(2)];
    nodes_coords[parent_cell.node(3)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(0, m_pattern - 1, 0)).node(3)];

    nodes_coords[parent_cell.node(4)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(0, 0, m_pattern - 1)).node(4)];
    nodes_coords[parent_cell.node(5)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(m_pattern - 1, 0, m_pattern - 1)).node(5)];
    nodes_coords[parent_cell.node(6)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(m_pattern - 1, m_pattern - 1, m_pattern - 1)).node(6)];
    nodes_coords[parent_cell.node(7)] = nodes_coords[childCellOfCell(parent_cell, CartCoord3(0, m_pattern - 1, m_pattern - 1)).node(7)];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal)
{
  // Pour avoir la face parent d'une maille, on passe d'abord de l'uid vers les
  // coordonnées de la maille,
  // puis on détermine les coordonnées du parent grâce au m_pattern,
  // et enfin, on repasse des coordonnées du parent vers son uid.

  if (globalNbCellsX(level - 1) == 0) {
    if (do_fatal) {
      ARCANE_FATAL("Level {0} do not exist", (level - 1));
    }
    return NULL_ITEM_UNIQUE_ID;
  }

  if (m_dimension == 2) {
    return cellUniqueId(CartCoord2(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level - 1),
                                        offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level - 1)),
                        level - 1);
  }

  return cellUniqueId(CartCoord3(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level - 1),
                                      offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level - 1),
                                      offsetLevelToLevel(cellUniqueIdToCoordZ(uid, level), level, level - 1)),
                      level - 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentCellUniqueIdOfCell(Cell cell, bool do_fatal)
{
  return parentCellUniqueIdOfCell(cell.uniqueId(), cell.level(), do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childCellUniqueIdOfCell(Cell cell, CartCoord3 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))
  ARCANE_ASSERT((child_coord_in_parent.z < m_pattern && child_coord_in_parent.z >= 0), ("Bad child_coord_in_parent.z"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return cellUniqueId(CartCoord3(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level + 1) + child_coord_in_parent.x,
                                      offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level + 1) + child_coord_in_parent.y,
                                      offsetLevelToLevel(cellUniqueIdToCoordZ(uid, level), level, level + 1) + child_coord_in_parent.z),
                      level + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childCellUniqueIdOfCell(Cell cell, CartCoord2 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  const Int64 uid = cell.uniqueId();
  const Int32 level = cell.level();

  return cellUniqueId(CartCoord2(offsetLevelToLevel(cellUniqueIdToCoordX(uid, level), level, level + 1) + child_coord_in_parent.x,
                                      offsetLevelToLevel(cellUniqueIdToCoordY(uid, level), level, level + 1) + child_coord_in_parent.y),
                      level + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent)
{
  if (m_dimension == 2) {
    ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

    return childCellUniqueIdOfCell(cell,
                                   CartCoord2(
                                   child_index_in_parent % m_pattern,
                                   child_index_in_parent / m_pattern));
  }

  ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern * m_pattern && child_index_in_parent >= 0), ("Bad child_index_in_parent"))

  const CartCoord to_2d = child_index_in_parent % (m_pattern * m_pattern);
  return childCellUniqueIdOfCell(cell,
                                 CartCoord3(
                                 to_2d % m_pattern,
                                 to_2d / m_pattern,
                                 child_index_in_parent / (m_pattern * m_pattern)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMngInternal::
childCellOfCell(Cell cell, CartCoord3 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  Cell child = cell.hChild(child_coord_in_parent.x + (child_coord_in_parent.y * m_pattern) + (child_coord_in_parent.z * m_pattern * m_pattern));
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMngInternal::
childCellOfCell(Cell cell, CartCoord2 child_coord_in_parent)
{
  ARCANE_ASSERT((child_coord_in_parent.x < m_pattern && child_coord_in_parent.x >= 0), ("Bad child_coord_in_parent.x"))
  ARCANE_ASSERT((child_coord_in_parent.y < m_pattern && child_coord_in_parent.y >= 0), ("Bad child_coord_in_parent.y"))

  Cell child = cell.hChild(child_coord_in_parent.x + (child_coord_in_parent.y * m_pattern));
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentNodeUniqueIdOfNode(Int64 uid, Int32 level, bool do_fatal)
{
  // Pour avoir le noeud parent d'un noeud, on passe d'abord de l'uid vers les
  // coordonnées du noeud, puis on détermine les coordonnées du parent grâce au m_pattern,
  // et enfin, on repasse des coordonnées du parent vers son uid.

  const CartCoord coord_x = nodeUniqueIdToCoordX(uid, level);
  const CartCoord coord_y = nodeUniqueIdToCoordY(uid, level);

  if (coord_x % m_pattern != 0 || coord_y % m_pattern != 0) {
    if (do_fatal) {
      ARCANE_FATAL("Node uid={0} do not have parent", uid);
    }
    return NULL_ITEM_UNIQUE_ID;
  }

  if (m_dimension == 2) {
    return nodeUniqueId(CartCoord2(offsetLevelToLevel(coord_x, level, level - 1),
                                        offsetLevelToLevel(coord_y, level, level - 1)),
                        level - 1);
  }

  const CartCoord coord_z = nodeUniqueIdToCoordZ(uid, level);

  if (coord_z % m_pattern != 0) {
    if (do_fatal) {
      ARCANE_FATAL("Node uid={0} do not have parent", uid);
    }
    return NULL_ITEM_UNIQUE_ID;
  }
  return nodeUniqueId(CartCoord3(offsetLevelToLevel(coord_x, level, level - 1),
                                      offsetLevelToLevel(coord_y, level, level - 1),
                                      offsetLevelToLevel(coord_z, level, level - 1)),
                      level - 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentNodeUniqueIdOfNode(Node node, bool do_fatal)
{
  const Int64 uid = node.uniqueId().asInt64();
  return parentNodeUniqueIdOfNode(uid, nodeLevel(uid), do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childNodeUniqueIdOfNode(Int64 uid, Int32 level)
{
  if (m_dimension == 2) {
    return nodeUniqueId(CartCoord2(offsetLevelToLevel(nodeUniqueIdToCoordX(uid, level), level, level + 1),
                                        offsetLevelToLevel(nodeUniqueIdToCoordY(uid, level), level, level + 1)),
                        level + 1);
  }

  return nodeUniqueId(CartCoord3(offsetLevelToLevel(nodeUniqueIdToCoordX(uid, level), level, level + 1),
                                      offsetLevelToLevel(nodeUniqueIdToCoordY(uid, level), level, level + 1),
                                      offsetLevelToLevel(nodeUniqueIdToCoordZ(uid, level), level, level + 1)),
                      level + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childNodeUniqueIdOfNode(Node node)
{
  const Int64 uid = node.uniqueId().asInt64();
  return childNodeUniqueIdOfNode(uid, nodeLevel(uid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentFaceUniqueIdOfFace(Int64 uid, Int32 level, bool do_fatal)
{
  if (m_converting_numbering_face && level == m_ori_level) {
    uid = m_face_ori_numbering_to_new[uid];
  }

  // Pour avoir la face parent d'une face, on passe d'abord de l'uid vers les
  // coordonnées de la face en "vue cartésienne",
  // puis on détermine les coordonnées du parent grâce au m_pattern,
  // et enfin, on repasse des coordonnées du parent vers son uid.

  const CartCoord coord_x = faceUniqueIdToCoordX(uid, level);
  const CartCoord coord_y = faceUniqueIdToCoordY(uid, level);

  ARCANE_ASSERT((coord_x < globalNbFacesXCartesianView(level) && coord_x >= 0), ("Bad coord_x"))
  ARCANE_ASSERT((coord_y < globalNbFacesYCartesianView(level) && coord_y >= 0), ("Bad coord_y"))

  const CartCoord parent_coord_x = faceOffsetLevelToLevel(coord_x, level, level - 1);
  const CartCoord parent_coord_y = faceOffsetLevelToLevel(coord_y, level, level - 1);

  if (parent_coord_x == -1 || parent_coord_y == -1) {
    if (do_fatal) {
      ARCANE_FATAL("Face uid={0} do not have parent", uid);
    }
    return NULL_ITEM_UNIQUE_ID;
  }

  ARCANE_ASSERT((parent_coord_x < globalNbFacesXCartesianView(level - 1) && parent_coord_x >= 0), ("Bad parent_coord_x"))
  ARCANE_ASSERT((parent_coord_y < globalNbFacesYCartesianView(level - 1) && parent_coord_y >= 0), ("Bad parent_coord_y"))

  if (m_dimension == 2) {
    if (m_converting_numbering_face && level - 1 == m_ori_level) {
      return m_face_new_numbering_to_ori[faceUniqueId(CartCoord2(parent_coord_x, parent_coord_y), level - 1)];
    }
    return faceUniqueId(CartCoord2(parent_coord_x, parent_coord_y), level - 1);
  }

  const CartCoord coord_z = faceUniqueIdToCoordZ(uid, level);
  ARCANE_ASSERT((coord_z < globalNbFacesZCartesianView(level) && coord_z >= 0), ("Bad coord_z"))

  const CartCoord parent_coord_z = faceOffsetLevelToLevel(coord_z, level, level - 1);

  if (parent_coord_z == -1) {
    if (do_fatal) {
      ARCANE_FATAL("Face uid={0} do not have parent", uid);
    }
    return NULL_ITEM_UNIQUE_ID;
  }

  ARCANE_ASSERT((parent_coord_z < globalNbFacesZCartesianView(level - 1) && parent_coord_z >= 0), ("Bad parent_coord_z"))

  //    debug() << "Uid : " << uid << " -- CoordX : " << coord_x << " -- CoordY : " << coord_y << " -- CoordZ : " << coord_z;

  if (m_converting_numbering_face && level - 1 == m_ori_level) {
    return m_face_new_numbering_to_ori[faceUniqueId(CartCoord3(parent_coord_x, parent_coord_y, parent_coord_z), level - 1)];
  }

  return faceUniqueId(CartCoord3(parent_coord_x, parent_coord_y, parent_coord_z), level - 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
parentFaceUniqueIdOfFace(Face face, bool do_fatal)
{
  const Int64 uid = face.uniqueId().asInt64();
  return parentFaceUniqueIdOfFace(uid, faceLevel(uid), do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childFaceUniqueIdOfFace(Int64 uid, Int32 level, Int32 child_index_in_parent)
{
  if (m_converting_numbering_face && level == m_ori_level) {
    uid = m_face_ori_numbering_to_new[uid];
  }

  const CartCoord coord_x = faceUniqueIdToCoordX(uid, level);
  const CartCoord coord_y = faceUniqueIdToCoordY(uid, level);

  ARCANE_ASSERT((coord_x < globalNbFacesXCartesianView(level) && coord_x >= 0), ("Bad coord_x"))
  ARCANE_ASSERT((coord_y < globalNbFacesYCartesianView(level) && coord_y >= 0), ("Bad coord_y"))

  CartCoord first_child_coord_x = faceOffsetLevelToLevel(coord_x, level, level + 1);
  CartCoord first_child_coord_y = faceOffsetLevelToLevel(coord_y, level, level + 1);

  ARCANE_ASSERT((first_child_coord_x < globalNbFacesXCartesianView(level + 1) && first_child_coord_x >= 0), ("Bad first_child_coord_x"))
  ARCANE_ASSERT((first_child_coord_y < globalNbFacesYCartesianView(level + 1) && first_child_coord_y >= 0), ("Bad first_child_coord_y"))

  if (m_dimension == 2) {
    ARCANE_ASSERT((child_index_in_parent < m_pattern && child_index_in_parent >= 0), ("Invalid child_index_in_parent"))

    if (coord_y % 2 == 0) {
      first_child_coord_x += child_index_in_parent * 2;
    }
    else if (coord_x % 2 == 0) {
      first_child_coord_y += child_index_in_parent * globalNbFacesY(level + 1);
    }
    else {
      ARCANE_FATAL("Impossible normalement");
    }

    if (m_converting_numbering_face && level + 1 == m_ori_level) {
      return m_face_new_numbering_to_ori[faceUniqueId(CartCoord2(first_child_coord_x, first_child_coord_y), level + 1)];
    }

    return faceUniqueId(CartCoord2(first_child_coord_x, first_child_coord_y), level + 1);
  }

  ARCANE_ASSERT((child_index_in_parent < m_pattern * m_pattern && child_index_in_parent >= 0), ("Invalid child_index_in_parent"))

  const CartCoord coord_z = faceUniqueIdToCoordZ(uid, level);
  ARCANE_ASSERT((coord_z < globalNbFacesZCartesianView(level) && coord_z >= 0), ("Bad coord_z"))

  CartCoord first_child_coord_z = faceOffsetLevelToLevel(coord_z, level, level + 1);
  ARCANE_ASSERT((first_child_coord_z < globalNbFacesZCartesianView(level + 1) && first_child_coord_z >= 0), ("Bad first_child_coord_z"))

  const CartCoord child_x = child_index_in_parent % m_pattern;
  const CartCoord child_y = child_index_in_parent / m_pattern;

  Int64x3 three_parts_numbering = _face3DNumberingThreeParts(level);

  if (uid < three_parts_numbering.x) {
    first_child_coord_y += child_x * 2;
    first_child_coord_z += child_y * 2;
  }
  else if (uid < three_parts_numbering.x + three_parts_numbering.y) {
    first_child_coord_x += child_x * 2;
    first_child_coord_z += child_y * 2;
  }
  else {
    first_child_coord_x += child_x * 2;
    first_child_coord_y += child_y * 2;
  }

  if (m_converting_numbering_face && level + 1 == m_ori_level) {
    return m_face_new_numbering_to_ori[faceUniqueId(CartCoord3(first_child_coord_x, first_child_coord_y, first_child_coord_z), level + 1)];
  }

  return faceUniqueId(CartCoord3(first_child_coord_x, first_child_coord_y, first_child_coord_z), level + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMngInternal::
childFaceUniqueIdOfFace(Face face, Int32 child_index_in_parent)
{
  const Int64 uid = face.uniqueId().asInt64();
  return childFaceUniqueIdOfFace(uid, faceLevel(uid), child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64x3 CartesianMeshNumberingMngInternal::
_face3DNumberingThreeParts(Int32 level) const
{
  const Int64x3 nb_cell(globalNbCellsX(level), globalNbCellsY(level), globalNbCellsZ(level));
  return { (nb_cell.x + 1) * nb_cell.y * nb_cell.z, (nb_cell.y + 1) * nb_cell.z * nb_cell.x, (nb_cell.z + 1) * nb_cell.x * nb_cell.y };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMngInternal::
_pushFront(UniqueArray<Int64>& array, const Int64 elem)
{
  array.resize(array.size() + 1);
  array.back() = elem;
  for (Integer i = array.size() - 2; i >= 0; --i) {
    std::swap(array[i], array[i + 1]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
