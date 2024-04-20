// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.h                                 (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"

#include "arcane/utils/TraceAccessor.h"

#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshNumberingMng
: public TraceAccessor
, public ICartesianMeshNumberingMng
{
 public:

  explicit CartesianMeshNumberingMng(IMesh* mesh);

 public:

  void prepareLevel(Int32 level) override;
  void updateFirstLevel() override;

  Int64 firstCellUniqueId(Integer level) override;
  Int64 firstNodeUniqueId(Integer level) override;
  Int64 firstFaceUniqueId(Integer level) override;

  Int64 globalNbCellsX(Integer level) const override;
  Int64 globalNbCellsY(Integer level) const override;
  Int64 globalNbCellsZ(Integer level) const override;

  Integer pattern() const override;
  Int64 offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const override;

  Int64 cellUniqueIdToCoordX(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordX(Cell cell) override;

  Int64 cellUniqueIdToCoordY(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordY(Cell cell) override;

  Int64 cellUniqueIdToCoordZ(Int64 uid, Integer level) override;
  Int64 cellUniqueIdToCoordZ(Cell cell) override;

  Int64 cellUniqueId(Integer level, Int64x3 cell_coord) override;
  Int64 cellUniqueId(Integer level, Int64x2 cell_coord) override;

  Integer nbNodeByCell() override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) override;
  void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;

  Integer nbFaceByCell() override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) override;
  void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;

  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) override;
  void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell) override;

  void setChildNodeCoordinates(Cell parent_cell) override;
  void setParentNodeCoordinates(Cell parent_cell) override;

  Int64 parentCellUniqueIdOfCell(Cell cell) override;

  Int64 childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent) override;
  Cell childCellOfCell(Cell cell, Int64x2 child_coord_in_parent) override;

  Int64 childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent) override;
  Cell childCellOfCell(Cell cell, Int64x3 child_coord_in_parent) override;

  Int64 childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent) override;

 private:
  IMesh* m_mesh;

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

  Int64x3 m_nb_cell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
