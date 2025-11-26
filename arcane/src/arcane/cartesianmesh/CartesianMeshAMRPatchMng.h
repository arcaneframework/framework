// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.h                                  (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/ICartesianMeshAMRPatchMng.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

#include "arcane/utils/TraceAccessor.h"

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

  explicit CartesianMeshAMRPatchMng(ICartesianMesh* cmesh, ICartesianMeshNumberingMng* numbering_mng);

 public:

  void flagCellToRefine(Int32ConstArrayView cells_lids, bool clear_old_flags) override;
  void flagCellToCoarsen(Int32ConstArrayView cells_lids, bool clear_old_flags) override;

  void refine() override;
  void createSubLevel() override;
  void coarsen(bool update_parent_flag) override;
  void _syncFlagCell() const override;

 private:

  void _shareInfosOfCellsAroundPatch(ConstArrayView<Cell> patch_cells, std::unordered_map<Int64, Integer>& around_cells_uid_to_owner, std::unordered_map<Int64, Int32>& around_cells_uid_to_flags, Int32 useful_flags) const;

 private:

  IMesh* m_mesh;
  ICartesianMesh* m_cmesh;
  ICartesianMeshNumberingMng* m_num_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRPATCHMNG_H
