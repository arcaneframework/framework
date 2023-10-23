// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.h                                  (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/cartesianmesh/ICartesianMeshAMRPatchMng.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianMeshAMRPatchMng
: public TraceAccessor
, public ICartesianMeshAMRPatchMng
{
 public:
  CartesianMeshAMRPatchMng(ICartesianMesh* mesh);
  void flagCellToRefine(Int32ConstArrayView cells_lids) override;
  void _syncFlagCell();
  void refine() override;
  void updateBackFrontCellFace();

 private:
  ICartesianMesh* m_cmesh;
  IMesh* m_mesh;
  CartesianMeshNumberingMng m_num_mng;
  Ref<VariableCellInteger> m_flag_cells_consistent;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
