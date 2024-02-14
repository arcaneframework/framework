// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.h                                  (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypedef.h"

#include "arcane/utils/TraceAccessor.h"

#include "arcane/cartesianmesh/ICartesianMeshAMRPatchMng.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshAMRPatchMng
: public TraceAccessor
, public ICartesianMeshAMRPatchMng
{
 public:
  explicit CartesianMeshAMRPatchMng(ICartesianMesh* mesh);

 public:
  void flagCellToRefine(Int32ConstArrayView cells_lids) override;
  void _syncFlagCell();
  void refine() override;

  void flagCellToCoarse(Int32ConstArrayView cells_lids) override;
  void coarse() override;

 private:
  IMesh* m_mesh;
  Ref<ICartesianMeshNumberingMng> m_num_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
