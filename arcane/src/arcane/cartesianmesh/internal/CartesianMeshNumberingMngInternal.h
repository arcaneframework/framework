// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMngInternal.h                         (C) 2000-2026 */
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

  void build() override;
  void saveInfosInProperties() override;
  void recreateFromDump() override;

  void renumberingFacesLevel0FromOriginalArcaneNumbering() override;

  void printStatus() override;

  void prepareLevel(Int32 level) override;
  void updateFirstLevel() override;

  Int64 firstCellUniqueId(Int32 level) const override;
  Int64 firstNodeUniqueId(Int32 level) const override;
  Int64 firstFaceUniqueId(Int32 level) const override;

  CartCoord globalNbCellsX(Int32 level) const override;
  CartCoord globalNbCellsY(Int32 level) const override;
  CartCoord globalNbCellsZ(Int32 level) const override;

  CartCoord globalNbNodesX(Int32 level) const override;
  CartCoord globalNbNodesY(Int32 level) const override;
  CartCoord globalNbNodesZ(Int32 level) const override;

  CartCoord globalNbFacesX(Int32 level) const override;
  CartCoord globalNbFacesY(Int32 level) const override;
  CartCoord globalNbFacesZ(Int32 level) const override;

  CartCoord globalNbFacesXCartesianView(Int32 level) const override;
  CartCoord globalNbFacesYCartesianView(Int32 level) const override;
  CartCoord globalNbFacesZCartesianView(Int32 level) const override;

  Int64 nbCellInLevel(Int32 level) const override;
  Int64 nbNodeInLevel(Int32 level) const override;
  Int64 nbFaceInLevel(Int32 level) const override;

  Int32 pattern() const override;

  Int32 cellLevel(Int64 uid) const override;
  Int32 nodeLevel(Int64 uid) const override;
  Int32 faceLevel(Int64 uid) const override;

  CartCoord offsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const override;
  CartCoord3 offsetLevelToLevel(CartCoord3 coord, Int32 level_from, Int32 level_to) const override;

  CartCoord faceOffsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const override;

  CartCoord3 cellUniqueIdToCoord(Int64 uid, Int32 level) override;
  CartCoord3 cellUniqueIdToCoord(Cell cell) override;

  CartCoord cellUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoord cellUniqueIdToCoordX(Cell cell) override;

  CartCoord cellUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoord cellUniqueIdToCoordY(Cell cell) override;

  CartCoord cellUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoord cellUniqueIdToCoordZ(Cell cell) override;

  CartCoord nodeUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoord nodeUniqueIdToCoordX(Node node) override;

  CartCoord nodeUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoord nodeUniqueIdToCoordY(Node node) override;

  CartCoord nodeUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoord nodeUniqueIdToCoordZ(Node node) override;

  CartCoord faceUniqueIdToCoordX(Int64 uid, Int32 level) override;
  CartCoord faceUniqueIdToCoordX(Face face) override;

  CartCoord faceUniqueIdToCoordY(Int64 uid, Int32 level) override;
  CartCoord faceUniqueIdToCoordY(Face face) override;

  CartCoord faceUniqueIdToCoordZ(Int64 uid, Int32 level) override;
  CartCoord faceUniqueIdToCoordZ(Face face) override;

  Int64 cellUniqueId(CartCoord3 cell_coord, Int32 level) override;
  Int64 cellUniqueId(CartCoord2 cell_coord, Int32 level) override;

  Int64 nodeUniqueId(CartCoord3 node_coord, Int32 level) override;
  Int64 nodeUniqueId(CartCoord2 node_coord, Int32 level) override;

  Int64 faceUniqueId(CartCoord3 face_coord, Int32 level) override;
  Int64 faceUniqueId(CartCoord2 face_coord, Int32 level) override;

  Int32 nbNodeByCell() override;
  void cellNodeUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid) override;

  Int32 nbFaceByCell() override;
  void cellFaceUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid) override;

  void cellUniqueIdsAroundCell(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid) override;

  void cellUniqueIdsAroundNode(CartCoord3 node_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(CartCoord2 node_coord, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid) override;
  void cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid) override;

  void setChildNodeCoordinates(Cell parent_cell) override;
  void setParentNodeCoordinates(Cell parent_cell) override;

  Int64 parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal) override;
  Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal) override;

  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord3 child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord2 child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent) override;

  Cell childCellOfCell(Cell cell, CartCoord3 child_coord_in_parent) override;
  Cell childCellOfCell(Cell cell, CartCoord2 child_coord_in_parent) override;

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

  CartCoord3 m_nb_cell_ground;

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
