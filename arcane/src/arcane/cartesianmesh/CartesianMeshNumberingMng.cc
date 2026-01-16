// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.cc                                (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

#include "arcane/utils/Vector2.h"
#include "arcane/utils/Vector3.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshNumberingMng::
CartesianMeshNumberingMng(ICartesianMesh* mesh)
: m_internal_api(mesh->_internalApi()->cartesianMeshNumberingMngInternal())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
printStatus() const
{
  m_internal_api->printStatus();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
firstCellUniqueId(Int32 level) const
{
  return m_internal_api->firstCellUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
firstNodeUniqueId(Int32 level) const
{
  return m_internal_api->firstNodeUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
firstFaceUniqueId(Int32 level) const
{
  return m_internal_api->firstFaceUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbCellsX(Int32 level) const
{
  return m_internal_api->globalNbCellsX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbCellsY(Int32 level) const
{
  return m_internal_api->globalNbCellsY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbCellsZ(Int32 level) const
{
  return m_internal_api->globalNbCellsZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbNodesX(Int32 level) const
{
  return m_internal_api->globalNbNodesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbNodesY(Int32 level) const
{
  return m_internal_api->globalNbNodesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbNodesZ(Int32 level) const
{
  return m_internal_api->globalNbNodesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesX(Int32 level) const
{
  return m_internal_api->globalNbFacesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesY(Int32 level) const
{
  return m_internal_api->globalNbFacesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesZ(Int32 level) const
{
  return m_internal_api->globalNbFacesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesXCartesianView(Int32 level) const
{
  return m_internal_api->globalNbFacesXCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesYCartesianView(Int32 level) const
{
  return m_internal_api->globalNbFacesYCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
globalNbFacesZCartesianView(Int32 level) const
{
  return m_internal_api->globalNbFacesZCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbCellInLevel(Int32 level) const
{
  return m_internal_api->nbCellInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbNodeInLevel(Int32 level) const
{
  return m_internal_api->nbNodeInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbFaceInLevel(Int32 level) const
{
  return m_internal_api->nbFaceInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
pattern() const
{
  return m_internal_api->pattern();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
cellLevel(Int64 uid) const
{
  return m_internal_api->cellLevel(uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
nodeLevel(Int64 uid) const
{
  return m_internal_api->nodeLevel(uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
faceLevel(Int64 uid) const
{
  return m_internal_api->faceLevel(uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
offsetLevelToLevel(CartCoordType coord, Int32 level_from, Int32 level_to) const
{
  return m_internal_api->offsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceOffsetLevelToLevel(CartCoordType coord, Int32 level_from, Int32 level_to) const
{
  return m_internal_api->faceOffsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordX(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordY(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordZ(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordX(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordY(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordX(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordY(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoordType CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordZ(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(CartCoord3Type cell_coord, Int32 level) const
{
  return m_internal_api->cellUniqueId(cell_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(CartCoord2Type cell_coord, Int32 level) const
{
  return m_internal_api->cellUniqueId(cell_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(CartCoord3Type node_coord, Int32 level) const
{
  return m_internal_api->nodeUniqueId(node_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(CartCoord2Type node_coord, Int32 level) const
{
  return m_internal_api->nodeUniqueId(node_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(CartCoord3Type face_coord, Int32 level) const
{
  return m_internal_api->faceUniqueId(face_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(CartCoord2Type face_coord, Int32 level) const
{
  return m_internal_api->faceUniqueId(face_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
nbNodeByCell() const
{
  return m_internal_api->nbNodeByCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellNodeUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellNodeUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellNodeUniqueIds(cell_uid, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid) const
{
  m_internal_api->cellNodeUniqueIds(cell, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshNumberingMng::
nbFaceByCell() const
{
  return m_internal_api->nbFaceByCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellFaceUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellFaceUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellFaceUniqueIds(cell_uid, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid) const
{
  m_internal_api->cellFaceUniqueIds(cell, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(CartCoord3Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundCell(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(CartCoord2Type cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundCell(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundCell(cell, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundNode(CartCoord3Type node_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundNode(node_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundNode(CartCoord2Type node_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundNode(node_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundNode(node_uid, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundNode(node, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundCell(cell_uid, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal) const
{
  return m_internal_api->parentCellUniqueIdOfCell(uid, level, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentCellUniqueIdOfCell(Cell cell, bool do_fatal) const
{
  return m_internal_api->parentCellUniqueIdOfCell(cell, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, CartCoord3Type child_coord_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, CartCoord2Type child_coord_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, CartCoord3Type child_coord_in_parent) const
{
  return m_internal_api->childCellOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, CartCoord2Type child_coord_in_parent) const
{
  return m_internal_api->childCellOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentNodeUniqueIdOfNode(Int64 uid, Int32 level, bool do_fatal) const
{
  return m_internal_api->parentNodeUniqueIdOfNode(uid, level, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentNodeUniqueIdOfNode(Node node, bool do_fatal) const
{
  return m_internal_api->parentNodeUniqueIdOfNode(node, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childNodeUniqueIdOfNode(Int64 uid, Int32 level) const
{
  return m_internal_api->childNodeUniqueIdOfNode(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childNodeUniqueIdOfNode(Node node) const
{
  return m_internal_api->childNodeUniqueIdOfNode(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentFaceUniqueIdOfFace(Int64 uid, Int32 level, bool do_fatal) const
{
  return m_internal_api->parentFaceUniqueIdOfFace(uid, level, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentFaceUniqueIdOfFace(Face face, bool do_fatal) const
{
  return m_internal_api->parentFaceUniqueIdOfFace(face, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childFaceUniqueIdOfFace(Int64 uid, Int32 level, Int32 child_index_in_parent) const
{
  return m_internal_api->childFaceUniqueIdOfFace(uid, level, child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childFaceUniqueIdOfFace(Face face, Int32 child_index_in_parent) const
{
  return m_internal_api->childFaceUniqueIdOfFace(face, child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICartesianMeshNumberingMngInternal* CartesianMeshNumberingMng::
_internalApi() const
{
  return m_internal_api.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
