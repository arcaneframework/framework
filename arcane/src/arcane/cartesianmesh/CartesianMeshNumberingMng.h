// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.h                                 (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICartesianMeshNumberingMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianMeshNumberingMng
{
 public:

  explicit CartesianMeshNumberingMng(ICartesianMesh* mesh);

 public:

  void printStatus() const;

  Int64 firstCellUniqueId(Integer level) const;
  Int64 firstNodeUniqueId(Integer level) const;
  Int64 firstFaceUniqueId(Integer level) const;

  Int64 globalNbCellsX(Integer level) const;
  Int64 globalNbCellsY(Integer level) const;
  Int64 globalNbCellsZ(Integer level) const;

  Int64 globalNbNodesX(Integer level) const;
  Int64 globalNbNodesY(Integer level) const;
  Int64 globalNbNodesZ(Integer level) const;

  Int64 globalNbFacesX(Integer level) const;
  Int64 globalNbFacesY(Integer level) const;
  Int64 globalNbFacesZ(Integer level) const;

  Int64 globalNbFacesXCartesianView(Integer level) const;
  Int64 globalNbFacesYCartesianView(Integer level) const;
  Int64 globalNbFacesZCartesianView(Integer level) const;

  Int64 nbCellInLevel(Integer level) const;
  Int64 nbNodeInLevel(Integer level) const;
  Int64 nbFaceInLevel(Integer level) const;

  Integer pattern() const;

  Int32 cellLevel(Int64 uid) const;
  Int32 nodeLevel(Int64 uid) const;
  Int32 faceLevel(Int64 uid) const;

  Int64 offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const;
  Int64 faceOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const;

  Int64 cellUniqueIdToCoordX(Int64 uid, Integer level) const;
  Int64 cellUniqueIdToCoordX(Cell cell) const;

  Int64 cellUniqueIdToCoordY(Int64 uid, Integer level) const;
  Int64 cellUniqueIdToCoordY(Cell cell) const;

  Int64 cellUniqueIdToCoordZ(Int64 uid, Integer level) const;
  Int64 cellUniqueIdToCoordZ(Cell cell) const;

  Int64 nodeUniqueIdToCoordX(Int64 uid, Integer level) const;
  Int64 nodeUniqueIdToCoordX(Node node) const;

  Int64 nodeUniqueIdToCoordY(Int64 uid, Integer level) const;
  Int64 nodeUniqueIdToCoordY(Node node) const;

  Int64 nodeUniqueIdToCoordZ(Int64 uid, Integer level) const;
  Int64 nodeUniqueIdToCoordZ(Node node) const;

  Int64 faceUniqueIdToCoordX(Int64 uid, Integer level) const;
  Int64 faceUniqueIdToCoordX(Face face) const;

  Int64 faceUniqueIdToCoordY(Int64 uid, Integer level) const;
  Int64 faceUniqueIdToCoordY(Face face) const;

  Int64 faceUniqueIdToCoordZ(Int64 uid, Integer level) const;
  Int64 faceUniqueIdToCoordZ(Face face) const;

  Int64 cellUniqueId(Integer level, Int64x3 cell_coord) const;
  Int64 cellUniqueId(Integer level, Int64x2 cell_coord) const;

  Int64 nodeUniqueId(Integer level, Int64x3 node_coord) const;
  Int64 nodeUniqueId(Integer level, Int64x2 node_coord) const;

  Int64 faceUniqueId(Integer level, Int64x3 face_coord) const;
  Int64 faceUniqueId(Integer level, Int64x2 face_coord) const;

  Integer nbNodeByCell() const;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) const;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) const;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) const;

  Integer nbFaceByCell() const;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) const;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) const;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) const;

  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) const;
  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell) const;

  Int64 parentCellUniqueIdOfCell(Int64 uid, Integer level, bool do_fatal) const;
  Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal) const;

  Int64 childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent) const;
  Int64 childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent) const;
  Int64 childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent) const;

  Cell childCellOfCell(Cell cell, Int64x3 child_coord_in_parent) const;
  Cell childCellOfCell(Cell cell, Int64x2 child_coord_in_parent) const;

  Int64 parentNodeUniqueIdOfNode(Int64 uid, Integer level, bool do_fatal) const;
  Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal) const;

  Int64 childNodeUniqueIdOfNode(Int64 uid, Integer level) const;
  Int64 childNodeUniqueIdOfNode(Node node) const;

  Int64 parentFaceUniqueIdOfFace(Int64 uid, Integer level, bool do_fatal) const;
  Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal) const;

  Int64 childFaceUniqueIdOfFace(Int64 uid, Integer level, Int64 child_index_in_parent) const;
  Int64 childFaceUniqueIdOfFace(Face face, Int64 child_index_in_parent) const;

 public:

  ICartesianMeshNumberingMngInternal* _internalApi() const;

 private:

  Ref<ICartesianMeshNumberingMngInternal> m_internal_api;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
