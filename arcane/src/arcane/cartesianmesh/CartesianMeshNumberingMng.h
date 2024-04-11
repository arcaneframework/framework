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

  CartesianMeshNumberingMng(IMesh* mesh);

 public:

  void prepareLevel(Int32 level) override;
  void updateFirstLevel() override;

  Int64 getFirstCellUidLevel(Integer level) override;
  Int64 getFirstNodeUidLevel(Integer level) override;
  Int64 getFirstFaceUidLevel(Integer level) override;

  Int64 getGlobalNbCellsX(Integer level) const override;
  Int64 getGlobalNbCellsY(Integer level) const override;
  Int64 getGlobalNbCellsZ(Integer level) const override;


  Integer getPattern() const override;
  Int64 getOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const override;


  Int64 uidToCoordX(Int64 uid, Integer level) override;
  Int64 uidToCoordX(Cell cell) override;

  Int64 uidToCoordY(Int64 uid, Integer level) override;
  Int64 uidToCoordY(Cell cell) override;

  Int64 uidToCoordZ(Int64 uid, Integer level) override;
  Int64 uidToCoordZ(Cell cell) override;


  Int64 getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) override;
  Int64 getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j) override;

  Integer getNbNode() override;
  void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) override;
  void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j) override;
  void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;

  Integer getNbFace() override;
  void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) override;
  void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j) override;
  void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid) override;

  void getCellUidsAround(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) override;
  void getCellUidsAround(ArrayView<Int64> uid, Cell cell) override;

  void setChildNodeCoordinates(Cell parent_cell) override;
  void setParentNodeCoordinates(Cell parent_cell) override;

  Int64 getParentCellUidOfCell(Cell cell) override;

  Int64 getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent) override;
  Cell getChildCellOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent) override;

  Int64 getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent, Int64 child_coord_z_in_parent) override;
  Cell getChildCellOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent, Int64 child_coord_z_in_parent) override;

  Int64 getChildCellUidOfCell(Cell cell, Int64 child_index_in_parent) override;

 private:
  IMesh* m_mesh;

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

  Int64 m_nb_cell_x;
  Int64 m_nb_cell_y;
  Int64 m_nb_cell_z;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
