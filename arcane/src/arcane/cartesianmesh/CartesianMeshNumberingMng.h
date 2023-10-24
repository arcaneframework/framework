// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.h                                 (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NumVector.h"

#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

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

  Integer getNbFace() override;
  void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) override;
  void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_cell_coord_i, Int64 cell_coord_j) override;

  void setNodeCoordinates(Cell child_cell) override;

 private:
  IMesh* m_mesh;

  Integer m_pattern;

  UniqueArray<Int64> m_first_cell_uid_level;
  UniqueArray<Int64> m_nb_cell_level;

  UniqueArray<Int64> m_first_node_uid_level;
  UniqueArray<Int64> m_nb_node_level;

  UniqueArray<Int64> m_first_face_uid_level;
  UniqueArray<Int64> m_nb_face_level;

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
