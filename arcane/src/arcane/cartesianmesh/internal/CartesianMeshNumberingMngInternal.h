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
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNGINTERNAL_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNGINTERNAL_H

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

  Int64 firstCellUniqueId(Integer level) const override;
  Int64 firstNodeUniqueId(Integer level) const override;
  Int64 firstFaceUniqueId(Integer level) const override;

  Int64 globalNbCellsX(Integer level) const override;
  Int64 globalNbCellsY(Integer level) const override;
  Int64 globalNbCellsZ(Integer level) const override;

  Int64 globalNbNodesX(Integer level) const override;
  Int64 globalNbNodesY(Integer level) const override;
  Int64 globalNbNodesZ(Integer level) const override;

  Int64 globalNbFacesX(Integer level) const override;
  Int64 globalNbFacesY(Integer level) const override;
  Int64 globalNbFacesZ(Integer level) const override;

  Int64 globalNbFacesXCartesianView(Integer level) const override;
  Int64 globalNbFacesYCartesianView(Integer level) const override;
  Int64 globalNbFacesZCartesianView(Integer level) const override;

  Int64 nbCellInLevel(Integer level) const override;
  Int64 nbNodeInLevel(Integer level) const override;
  Int64 nbFaceInLevel(Integer level) const override;

  Integer pattern() const override;

  Int32 cellLevel(Int64 uid) const override;
  Int32 nodeLevel(Int64 uid) const override;
  Int32 faceLevel(Int64 uid) const override;

  Int64 offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const override;
  Int64 faceOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const override;

  Int64 cellUniqueIdToCoordX(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordX(Cell cell) override;

  Int64 cellUniqueIdToCoordY(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordY(Cell cell) override;

  Int64 cellUniqueIdToCoordZ(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordZ(Cell cell) override;

  Int64 nodeUniqueIdToCoordX(Int64 uid, Integer level) override;
  Int64 nodeUniqueIdToCoordX(Node node) override;

  Int64 nodeUniqueIdToCoordY(Int64 uid, Integer level) override;
  Int64 nodeUniqueIdToCoordY(Node node) override;

  Int64 nodeUniqueIdToCoordZ(Int64 uid, Integer level) override;
  Int64 nodeUniqueIdToCoordZ(Node node) override;

  Int64 faceUniqueIdToCoordX(Int64 uid, Integer level) override;
  Int64 faceUniqueIdToCoordX(Face face) override;

  Int64 faceUniqueIdToCoordY(Int64 uid, Integer level) override;
  Int64 faceUniqueIdToCoordY(Face face) override;

  Int64 faceUniqueIdToCoordZ(Int64 uid, Integer level) override;
  Int64 faceUniqueIdToCoordZ(Face face) override;

  Int64 cellUniqueId(Integer level, Int64x3 cell_coord) override;
  Int64 cellUniqueId(Integer level, Int64x2 cell_coord) override;

  Int64 nodeUniqueId(Integer level, Int64x3 node_coord) override;
  Int64 nodeUniqueId(Integer level, Int64x2 node_coord) override;

  Int64 faceUniqueId(Integer level, Int64x3 face_coord) override;
  Int64 faceUniqueId(Integer level, Int64x2 face_coord) override;

  Integer nbNodeByCell() override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Cell cell) override;

  Integer nbFaceByCell() override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Cell cell) override;

  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64x3 cell_coord, Int32 level) override;
  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64x2 cell_coord, Int32 level) override;
  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) override;
  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell) override;

  void cellUniqueIdsAroundNode(ArrayView<Int64> uid, Int64x3 node_coord, Int32 level) override;
  void cellUniqueIdsAroundNode(ArrayView<Int64> uid, Int64x2 node_coord, Int32 level) override;
  void cellUniqueIdsAroundNode(ArrayView<Int64> uid, Int64 node_uid, Int32 level) override;
  void cellUniqueIdsAroundNode(ArrayView<Int64> uid, Node node) override;

  void setChildNodeCoordinates(Cell parent_cell) override;
  void setParentNodeCoordinates(Cell parent_cell) override;

  Int64 parentCellUniqueIdOfCell(Int64 uid, Integer level, bool do_fatal) override;
  Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal) override;

  Int64 childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent) override;
  Int64 childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent) override;

  Cell childCellOfCell(Cell cell, Int64x3 child_coord_in_parent) override;
  Cell childCellOfCell(Cell cell, Int64x2 child_coord_in_parent) override;

  Int64 parentNodeUniqueIdOfNode(Int64 uid, Integer level, bool do_fatal) override;
  Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal) override;

  Int64 childNodeUniqueIdOfNode(Int64 uid, Integer level) override;
  Int64 childNodeUniqueIdOfNode(Node node) override;

  Int64 parentFaceUniqueIdOfFace(Int64 uid, Integer level, bool do_fatal) override;
  Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal) override;

  Int64 childFaceUniqueIdOfFace(Int64 uid, Integer level, Int64 child_index_in_parent) override;
  Int64 childFaceUniqueIdOfFace(Face face, Int64 child_index_in_parent) override;

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
  Int64x3 _face3DNumberingThreeParts(Integer level) const;

  static void _pushFront(UniqueArray<Int64>& array, Int64 elem);

 private:

  IMesh* m_mesh;

  Ref<Properties> m_properties;

  Integer m_dimension;
  Integer m_pattern;

  UniqueArray<Int32> m_p_to_l_level;
  Int32 m_max_level;
  Int32 m_min_level;

  Int64 m_latest_cell_uid;
  UniqueArray<Int64> m_first_cell_uid_level;

  Int64 m_latest_node_uid;
  UniqueArray<Int64> m_first_node_uid_level;

  Int64 m_latest_face_uid;
  UniqueArray<Int64> m_first_face_uid_level;

  Int64x3 m_nb_cell_ground;

  // Partie conversion numérotation d'origine <-> nouvelle numérotation (face).
  bool m_converting_numbering_face;
  Integer m_ori_level;
  std::unordered_map<Int64, Int64> m_face_ori_numbering_to_new;
  std::unordered_map<Int64, Int64> m_face_new_numbering_to_ori;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNGINTERNAL_H
