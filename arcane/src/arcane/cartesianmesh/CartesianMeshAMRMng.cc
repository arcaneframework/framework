// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"
#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"
#include "arcane/cartesianmesh/CartesianMeshUtils.h"
#include "arcane/cartesianmesh/CartesianPatch.h"

#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/MeshKind.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAMRMng::
CartesianMeshAMRMng(ICartesianMesh* cmesh)
: m_cmesh(cmesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAMRMng::
nbPatch() const
{
  return m_cmesh->nbPatch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianPatch CartesianMeshAMRMng::
amrPatch(Int32 index) const
{
  return m_cmesh->amrPatch(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshPatchListView CartesianMeshAMRMng::
patches() const
{
  return m_cmesh->patches();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
refineZone(const AMRZonePosition& position) const
{
  m_cmesh->refinePatch(position);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
coarseZone(const AMRZonePosition& position) const
{
  m_cmesh->coarseZone(position);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
refine() const
{
  m_cmesh->refine();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianMeshAMRMng::
reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers) const
{
  return m_cmesh->reduceNbGhostLayers(level, target_nb_ghost_layers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
mergePatches() const
{
  m_cmesh->mergePatches();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
createSubLevel() const
{
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if(amr_type == eMeshAMRKind::Cell) {
    Ref<CartesianMeshCoarsening2> coarser = CartesianMeshUtils::createCartesianMeshCoarsening2(m_cmesh);
    coarser->createCoarseCells();
  }
  else if(amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_cmesh->_internalApi()->cartesianMeshAMRPatchMng()->createSubLevel();
  }
  else if(amr_type == eMeshAMRKind::Patch) {
    ARCANE_FATAL("General patch AMR is not implemented. Please use PatchCartesianMeshOnly (3)");
  }
  else{
    ARCANE_FATAL("AMR is not enabled");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
