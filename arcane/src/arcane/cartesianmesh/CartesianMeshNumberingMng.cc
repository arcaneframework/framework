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

CartCoord CartesianMeshNumberingMng::
globalNbCellsX(Int32 level) const
{
  return m_internal_api->globalNbCellsX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbCellsY(Int32 level) const
{
  return m_internal_api->globalNbCellsY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbCellsZ(Int32 level) const
{
  return m_internal_api->globalNbCellsZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbNodesX(Int32 level) const
{
  return m_internal_api->globalNbNodesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbNodesY(Int32 level) const
{
  return m_internal_api->globalNbNodesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbNodesZ(Int32 level) const
{
  return m_internal_api->globalNbNodesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbFacesX(Int32 level) const
{
  return m_internal_api->globalNbFacesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbFacesY(Int32 level) const
{
  return m_internal_api->globalNbFacesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbFacesZ(Int32 level) const
{
  return m_internal_api->globalNbFacesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbFacesXCartesianView(Int32 level) const
{
  return m_internal_api->globalNbFacesXCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
globalNbFacesYCartesianView(Int32 level) const
{
  return m_internal_api->globalNbFacesYCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
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

CartCoord CartesianMeshNumberingMng::
offsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const
{
  return m_internal_api->offsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceOffsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const
{
  return m_internal_api->faceOffsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordX(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordY(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->cellUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordZ(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordX(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordY(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordX(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordY(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Int64 uid, Int32 level) const
{
  return m_internal_api->faceUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordZ(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(CartCoord3 cell_coord, Int32 level) const
{
  return m_internal_api->cellUniqueId(cell_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(CartCoord2 cell_coord, Int32 level) const
{
  return m_internal_api->cellUniqueId(cell_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(CartCoord3 node_coord, Int32 level) const
{
  return m_internal_api->nodeUniqueId(node_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(CartCoord2 node_coord, Int32 level) const
{
  return m_internal_api->nodeUniqueId(node_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(CartCoord3 face_coord, Int32 level) const
{
  return m_internal_api->faceUniqueId(face_coord, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(CartCoord2 face_coord, Int32 level) const
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
cellNodeUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellNodeUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const
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
cellFaceUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellFaceUniqueIds(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const
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
cellUniqueIdsAroundCell(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundCell(cell_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const
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
cellUniqueIdsAroundNode(CartCoord3 node_coord, Int32 level, ArrayView<Int64> uid) const
{
  m_internal_api->cellUniqueIdsAroundNode(node_coord, level, uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundNode(CartCoord2 node_coord, Int32 level, ArrayView<Int64> uid) const
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
childCellUniqueIdOfCell(Cell cell, CartCoord3 child_coord_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, CartCoord2 child_coord_in_parent) const
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
childCellOfCell(Cell cell, CartCoord3 child_coord_in_parent) const
{
  return m_internal_api->childCellOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, CartCoord2 child_coord_in_parent) const
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
