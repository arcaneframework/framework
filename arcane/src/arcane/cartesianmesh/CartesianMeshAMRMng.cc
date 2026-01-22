// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Gestionnaire de l'AMR pour un maillage cartésien.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/MeshKind.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianPatch.h"
#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"
#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"
#include "arcane/cartesianmesh/CartesianMeshUtils.h"

#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"
#include "arcane/cartesianmesh/internal/CartesianPatchGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAMRMng::
CartesianMeshAMRMng(ICartesianMesh* cmesh)
: m_cmesh(cmesh)
{}

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
beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)
{
  // On calcule le nombre de mailles de recouvrements pour chaque level.
  m_cmesh->_internalApi()->cartesianPatchGroup().beginAdaptMesh(max_nb_levels, level_to_refine_first);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
adaptLevel(Int32 level_to_adapt) const
{
  m_cmesh->_internalApi()->cartesianPatchGroup().adaptLevel(level_to_adapt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
endAdaptMesh()
{
  m_cmesh->_internalApi()->cartesianPatchGroup().endAdaptMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
clearRefineRelatedFlags() const
{
  m_cmesh->_internalApi()->cartesianPatchGroup().clearRefineRelatedFlags();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
setOverlapLayerSizeTopLevel(Int32 new_size) const
{
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if (amr_type == eMeshAMRKind::Cell) {
    return;
  }
  m_cmesh->_internalApi()->cartesianPatchGroup().setOverlapLayerSizeTopLevel(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRMng::
disableOverlapLayer()
{
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if (amr_type == eMeshAMRKind::Cell) {
    return;
  }
  m_cmesh->_internalApi()->cartesianPatchGroup().setOverlapLayerSizeTopLevel(-1);
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
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if (amr_type == eMeshAMRKind::Cell) {
    return;
  }

  m_cmesh->_internalApi()->cartesianPatchGroup().mergePatches();
  m_cmesh->_internalApi()->cartesianPatchGroup().applyPatchEdit(false);
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
