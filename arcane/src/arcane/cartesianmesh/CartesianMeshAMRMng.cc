// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.cc                                      (C) 2000-2025 */
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
adaptMesh() const
{
  /*
   * Dans le cas d'un raffinement niveau par niveau, il est possible de mettre
   * le paramètre \a clear_refine_flag à false afin de garder les flags des
   * niveaux inférieurs et d'éviter d'avoir à les recalculer. Pour le dernier
   * niveau, il est recommandé de mettre le paramètre \a clear_refine_flag à
   * true pour supprimer les flags devenu inutiles (ou d'appeler la méthode
   * clearRefineRelatedFlags()).
   *
   * Les mailles n'ayant pas de flag "II_Refine" seront déraffinées.
   *
   * Afin d'éviter les mailles orphelines, si une maille est marquée
   * "II_Refine", alors la maille parente est marquée "II_Refine".
   *
   * Exemple d'exécution :
   * ```
   * CartesianMeshAMRMng amr_mng(cmesh());
   * amr_mng.clearRefineRelatedFlags();
   * for (Integer level = 0; level < 2; ++level){
   *   computeInLevel(level); // Va mettre des flags II_Refine sur les mailles
   *   amr_mng.adaptMesh(false);
   * }
   * amr_mng.clearRefineRelatedFlags();
   * ```
   *
   * Cette opération est collective.
   *
   * \param clear_refine_flag true si l'on souhaite supprimer les flags
   * II_Refine après adaptation.
   */
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if (amr_type == eMeshAMRKind::Cell) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  m_cmesh->_internalApi()->cartesianPatchGroup().refine(true);
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
enableOverlapLayer(bool enable) const
{
  auto amr_type = m_cmesh->mesh()->meshKind().meshAMRKind();
  if (amr_type == eMeshAMRKind::Cell) {
    return;
  }
  m_cmesh->_internalApi()->cartesianPatchGroup().setOverlapLayerSizeTopLevel(enable ? 2 : 0);
  m_cmesh->computeDirections();
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
