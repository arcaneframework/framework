// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMngInternal.h                         (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* des mailles et des noeuds est assez classique, la numérotation des faces  */
/* est expliquée (entre autres) dans les méthodes 'faceUniqueId()' et        */
/* 'cellFaceUniqueIds()'.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_INTERNAL_CARTESIANMESHNUMBERINGMNGINTERNAL_H
#define ARCANE_CARTESIANMESH_INTERNAL_CARTESIANMESHNUMBERINGMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/ICartesianMeshNumberingMngInternal.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Vector3.h"

#include "arcane/core/Item.h"

#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshNumberingMngInternal
: public TraceAccessor
, public ICartesianMeshNumberingMngInternal
{
 public:

  explicit CartesianMeshNumberingMngInternal(IMesh* mesh);

 public:

  void _build() override;
  void _saveInfosInProperties() override;
  void _recreateFromDump() override;

  void renumberingFacesLevel0FromOriginalArcaneNumbering() override;

  void printStatus() override;

  void prepareLevel(Int32 level) override;
  void updateFirstLevel() override;

  Int64 firstCellUniqueId(Int32 level) const override;
  Int64 firstNodeUniqueId(Int32 level) const override;
  Int64 firstFaceUniqueId(Int32 level) const override;

  CartCoordType globalNbCellsX(Int32 level) const override;
  CartCoordType globalNbCellsY(Int32 level) const override;
  CartCoordType globalNbCellsZ(Int32 level) const override;

  CartCoordType globalNbNodesX(Int32 level) const override;
  CartCoordType globalNbNodesY(Int32 level) const override;
  CartCoordType globalNbNodesZ(Int32 level) const override;

  CartCoordType globalNbFacesX(Int32 level) const override;
  CartCoordType globalNbFacesY(Int32 level) const override;
  CartCoordType globalNbFacesZ(Int32 level) const override;

  CartCoordType globalNbFacesXCartesianView(Int32 level) const override;
  CartCoordType globalNbFacesYCartesianView(Int32 level) const override;
  CartCoordType globalNbFacesZCartesianView(Int32 level) const override;

  Int64 nbCellInLevel(Int32 level) const override;
  Int64 nbNodeInLevel(Int32 level) const override;
  Int64 nbFaceInLevel(Int32 level) const override;

  Int32 pattern() const override;

  Int32 cellLevel(Int64 uid) const override;
  Int32 nodeLevel(Int64 uid) const override;
  Int32 faceLevel(Int64 uid) const override;

  CartCoordType offsetLevelToLevel(CartCoordType coord, Int32 level_from, Int32 level_to) const override;
  CartCoord3Type offsetLevelToLevel(CartCoord3Type coord, Int32 level_from, Int32 level_to) const override;

  CartCoordType faceOffsetLevelToLevel(CartCoordType coord, Int32 level_from, Int32 level_to) const override;

  CartCoord3Type cellUniqueIdToCoord(Int64 uid, Int32 level) override;
  CartCoord3Type cellUniqueIdToCoord(Cell cell) override;

  CartCoordType cellUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoordType cellUniqueIdToCoordX(Cell cell) override;

  CartCoordType cellUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoordType cellUniqueIdToCoordY(Cell cell) override;

  CartCoordType cellUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoordType cellUniqueIdToCoordZ(Cell cell) override;

  CartCoordType nodeUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoordType nodeUniqueIdToCoordX(Node node) override;

  CartCoordType nodeUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoordType nodeUniqueIdToCoordY(Node node) override;

  CartCoordType nodeUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoordType nodeUniqueIdToCoordZ(Node node) override;

  CartCoordType faceUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoordType faceUniqueIdToCoordX(Face face) override;

  CartCoordType faceUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoordType faceUniqueIdToCoordY(Face face) override;

  CartCoordType faceUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoordType faceUniqueIdToCoordZ(Face face) override;

  Int64 cellUniqueId(CartCoord3Type cell_coord, Int32 level) override;
  Int64 cellUniqueId(CartCoord2Type cell_coord, Int32 level) override;

  Int64 nodeUniqueId(CartCoord3Type node_coord, Int32 level) override;
  Int64 nodeUniqueId(CartCoord2Type node_coord, Int32 level) override;

  Int64 faceUniqueId(CartCoord3Type face_coord, Int32 level) override;
  Int64 faceUniqueId(CartCoord2Type face_coord, Int32 level) override;

  Int32 nbNodeByCell() override;
  void cellNodeUniqueIds(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid) override;

  Int32 nbFaceByCell() override;
  void cellFaceUniqueIds(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid) override;

  void cellUniqueIdsAroundCell(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid) override;

  void cellUniqueIdsAroundNode(CartCoord3Type node_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(CartCoord2Type node_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid) override;

  void setChildNodeCoordinates(Cell parent_cell) override;
  void setParentNodeCoordinates(Cell parent_cell) override;

  Int64 parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal) override;
  Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal) override;

  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord3Type child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord2Type child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent) override;

  Cell childCellOfCell(Cell cell, CartCoord3Type child_coord_in_parent) override;
  Cell childCellOfCell(Cell cell, CartCoord2Type child_coord_in_parent) override;

  Int64 parentNodeUniqueIdOfNode(Int64 uid, Int32 level, bool do_fatal) override;
  Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal) override;

  Int64 childNodeUniqueIdOfNode(Int64 uid, Int32 level) override;
  Int64 childNodeUniqueIdOfNode(Node node) override;

  Int64 parentFaceUniqueIdOfFace(Int64 uid, Int32 level, bool do_fatal) override;
  Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal) override;

  Int64 childFaceUniqueIdOfFace(Int64 uid, Int32 level, Int32 child_index_in_parent) override;
  Int64 childFaceUniqueIdOfFace(Face face, Int32 child_index_in_parent) override;

 private:

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces des trois parties de la numérotation.
   *
   * En effet, pour numéroter en 3D, on numérote d'abord les faces xy, puis les faces yz et enfin
   * les faces zx. Cette méthode permet de récupérer le nombre de faces {xy, yz, zx}.
   *
   * \param level Le niveau de la numérotation.
   * \return Le nombre de faces {xy, yz, zx}.
   */
  Int64x3 _face3DNumberingThreeParts(Int32 level) const;

  static void _pushFront(UniqueArray<Int64>& array, Int64 elem);

 private:

  IMesh* m_mesh;

  Ref<Properties> m_properties;

  Integer m_dimension;
  Int32 m_pattern;

  UniqueArray<Int32> m_p_to_l_level;
  Int32 m_max_level;
  Int32 m_min_level;

  Int64 m_latest_cell_uid;
  UniqueArray<Int64> m_first_cell_uid_level;

  Int64 m_latest_node_uid;
  UniqueArray<Int64> m_first_node_uid_level;

  Int64 m_latest_face_uid;
  UniqueArray<Int64> m_first_face_uid_level;

  CartCoord3Type m_nb_cell_ground;

  // Partie conversion numérotation d'origine <-> nouvelle numérotation (face).
  bool m_converting_numbering_face;
  Int32 m_ori_level;
  std::unordered_map<Int64, Int64> m_face_ori_numbering_to_new;
  std::unordered_map<Int64, Int64> m_face_new_numbering_to_ori;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNGINTERNAL_H
