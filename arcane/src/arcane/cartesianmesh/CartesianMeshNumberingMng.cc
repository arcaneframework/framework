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
{
}

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
firstCellUniqueId(Integer level) const
{
  return m_internal_api->firstCellUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
firstNodeUniqueId(Integer level) const
{
  return m_internal_api->firstNodeUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
firstFaceUniqueId(Integer level) const
{
  return m_internal_api->firstFaceUniqueId(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbCellsX(Integer level) const
{
  return m_internal_api->globalNbCellsX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbCellsY(Integer level) const
{
  return m_internal_api->globalNbCellsY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbCellsZ(Integer level) const
{
  return m_internal_api->globalNbCellsZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbNodesX(Integer level) const
{
  return m_internal_api->globalNbNodesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbNodesY(Integer level) const
{
  return m_internal_api->globalNbNodesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbNodesZ(Integer level) const
{
  return m_internal_api->globalNbNodesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesX(Integer level) const
{
  return m_internal_api->globalNbFacesX(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesY(Integer level) const
{
  return m_internal_api->globalNbFacesY(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesZ(Integer level) const
{
  return m_internal_api->globalNbFacesZ(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesXCartesianView(Integer level) const
{
  return m_internal_api->globalNbFacesXCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesYCartesianView(Integer level) const
{
  return m_internal_api->globalNbFacesYCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
globalNbFacesZCartesianView(Integer level) const
{
  return m_internal_api->globalNbFacesZCartesianView(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbCellInLevel(Integer level) const
{
  return m_internal_api->nbCellInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbNodeInLevel(Integer level) const
{
  return m_internal_api->nbNodeInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nbFaceInLevel(Integer level) const
{
  return m_internal_api->nbFaceInLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianMeshNumberingMng::
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

Int64 CartesianMeshNumberingMng::
offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const
{
  return m_internal_api->offsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const
{
  return m_internal_api->faceOffsetLevelToLevel(coord, level_from, level_to);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Int64 uid, Integer level) const
{
  return m_internal_api->cellUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordX(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordX(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Int64 uid, Integer level) const
{
  return m_internal_api->cellUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordY(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordY(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Int64 uid, Integer level) const
{
  return m_internal_api->cellUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueIdToCoordZ(Cell cell) const
{
  return m_internal_api->cellUniqueIdToCoordZ(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Int64 uid, Integer level) const
{
  return m_internal_api->nodeUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordX(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordX(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Int64 uid, Integer level) const
{
  return m_internal_api->nodeUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordY(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordY(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Int64 uid, Integer level) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueIdToCoordZ(Node node) const
{
  return m_internal_api->nodeUniqueIdToCoordZ(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Int64 uid, Integer level) const
{
  return m_internal_api->faceUniqueIdToCoordX(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordX(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordX(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Int64 uid, Integer level) const
{
  return m_internal_api->faceUniqueIdToCoordY(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordY(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordY(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Int64 uid, Integer level) const
{
  return m_internal_api->faceUniqueIdToCoordZ(uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueIdToCoordZ(Face face) const
{
  return m_internal_api->faceUniqueIdToCoordZ(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(Integer level, Int64x3 cell_coord) const
{
  return m_internal_api->cellUniqueId(level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
cellUniqueId(Integer level, Int64x2 cell_coord) const
{
  return m_internal_api->cellUniqueId(level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(Integer level, Int64x3 node_coord) const
{
  return m_internal_api->nodeUniqueId(level, node_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
nodeUniqueId(Integer level, Int64x2 node_coord) const
{
  return m_internal_api->nodeUniqueId(level, node_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(Integer level, Int64x3 face_coord) const
{
  return m_internal_api->faceUniqueId(level, face_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
faceUniqueId(Integer level, Int64x2 face_coord) const
{
  return m_internal_api->faceUniqueId(level, face_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianMeshNumberingMng::
nbNodeByCell() const
{
  return m_internal_api->nbNodeByCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) const
{
  m_internal_api->cellNodeUniqueIds(uid, level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) const
{
  m_internal_api->cellNodeUniqueIds(uid, level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) const
{
  m_internal_api->cellNodeUniqueIds(uid, level, cell_uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianMeshNumberingMng::
nbFaceByCell() const
{
  return m_internal_api->nbFaceByCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) const
{
  m_internal_api->cellFaceUniqueIds(uid, level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) const
{
  m_internal_api->cellFaceUniqueIds(uid, level, cell_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) const
{
  m_internal_api->cellFaceUniqueIds(uid, level, cell_uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell) const
{
  m_internal_api->cellUniqueIdsAroundCell(uid, cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshNumberingMng::
cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) const
{
  m_internal_api->cellUniqueIdsAroundCell(uid, cell_uid, level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentCellUniqueIdOfCell(Int64 uid, Integer level, bool do_fatal) const
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
childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent) const
{
  return m_internal_api->childCellUniqueIdOfCell(cell, child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, Int64x3 child_coord_in_parent) const
{
  return m_internal_api->childCellOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CartesianMeshNumberingMng::
childCellOfCell(Cell cell, Int64x2 child_coord_in_parent) const
{
  return m_internal_api->childCellOfCell(cell, child_coord_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
parentNodeUniqueIdOfNode(Int64 uid, Integer level, bool do_fatal) const
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
childNodeUniqueIdOfNode(Int64 uid, Integer level) const
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
parentFaceUniqueIdOfFace(Int64 uid, Integer level, bool do_fatal) const
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
childFaceUniqueIdOfFace(Int64 uid, Integer level, Int64 child_index_in_parent) const
{
  return m_internal_api->childFaceUniqueIdOfFace(uid, level, child_index_in_parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshNumberingMng::
childFaceUniqueIdOfFace(Face face, Int64 child_index_in_parent) const
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
