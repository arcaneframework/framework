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

  CartesianMeshAMRPatchMng(IMesh* mesh);
  void flagCellToRefine(Int32ConstArrayView cells_lids) override;
  void refine() override;

 private:

  IMesh* m_mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
