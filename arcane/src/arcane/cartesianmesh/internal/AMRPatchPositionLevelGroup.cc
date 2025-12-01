// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/AMRPatchPositionLevelGroup.h"
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionLevelGroup::
AMRPatchPositionLevelGroup(Integer max_level)
: m_max_level(max_level)
, m_patches(max_level+1)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionLevelGroup::
~AMRPatchPositionLevelGroup()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer AMRPatchPositionLevelGroup::
maxLevel()
{
  return m_max_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<AMRPatchPosition> AMRPatchPositionLevelGroup::
patches(Integer level)
{
  return m_patches[level];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionLevelGroup::
addPatch(AMRPatchPosition patch)
{
  m_patches[patch.level()].add(patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionLevelGroup::
fusionPatches(Integer level)
{
  fusionPatches(m_patches[level], true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionLevelGroup::
fusionPatches(UniqueArray<AMRPatchPosition>& patch_position, bool remove_null)
{
  // Algo de fusion.
  // D'abord, on trie les patchs du plus petit nb de mailles au plus grand nb de mailles (optionnel).
  // Ensuite, pour chaque patch, on regarde si l'on peut le fusionner avec un autre.
  // Si on arrive à faire une fusion, on recommence l'algo jusqu'à ne plus pouvoir fusionner.
  bool fusion = true;
  while (fusion) {
    fusion = false;

    std::sort(patch_position.begin(), patch_position.end(),
              [](const AMRPatchPosition& a, const AMRPatchPosition& b) {
                return a.nbCells() < b.nbCells();
              });

    for (Integer p0 = 0; p0 < patch_position.size(); ++p0) {
      AMRPatchPosition& patch_fusion_0 = patch_position[p0];
      if (patch_fusion_0.isNull())
        continue;

      // Si une fusion a déjà eu lieu, on doit alors regarder les patchs avant "p0"
      // (vu qu'il y en a au moins un qui a été modifié).
      // (une "optimisation" pourrait être de récupérer la position du premier
      // patch fusionné mais bon, moins lisible + pas beaucoup de patchs).
      Integer p1 = (fusion ? 0 : p0 + 1);
      for (; p1 < patch_position.size(); ++p1) {
        if (p1 == p0)
          continue;

        AMRPatchPosition& patch_fusion_1 = patch_position[p1];

        if (patch_fusion_1.isNull())
          continue;

        // m_cmesh->traceMng()->info() << "\tCheck fusion"
        //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
        //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
        //                                     << " -- 0 Level : " << patch_fusion_0.level()
        //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
        //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
        //                                     << " -- 1 Level : " << patch_fusion_1.level();
        if (patch_fusion_0.canBeFusion(patch_fusion_1)) {
          // m_cmesh->traceMng()->info() << "Fusion OK";
          patch_fusion_0.fusion(patch_fusion_1);
          patch_fusion_1.setLevel(-2); // Devient null.
          fusion = true;
          break;
        }
      }
    }
  }
  if (remove_null) {
    for (Integer i = 0; i < patch_position.size(); ++i) {
      if (patch_position[i].isNull()) {
        patch_position.remove(i);
        i--;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
