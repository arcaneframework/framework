// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.cc                               (C) 2000-2026 */
/*                                                                           */
/* Groupe de position de patch AMR réparti par niveau.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/AMRPatchPositionLevelGroup.h"

#include "arcane/cartesianmesh/AMRPatchPosition.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionLevelGroup::
AMRPatchPositionLevelGroup(Int32 max_level)
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

Int32 AMRPatchPositionLevelGroup::
maxLevel() const
{
  return m_max_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<AMRPatchPosition> AMRPatchPositionLevelGroup::
patches(Int32 level)
{
  return m_patches[level];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionLevelGroup::
addPatch(const AMRPatchPosition& patch)
{
  m_patches[patch.level()].add(patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionLevelGroup::
fusionPatches(Int32 level)
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

    // Le sort permet de fusionner les patchs les plus petits d'abord, ce qui
    // permet d'équilibrer (un peu) la taille des patchs.
    // Attention : retirer le sort change les hashs des tests.
    std::stable_sort(patch_position.begin(), patch_position.end(),
                     [](const AMRPatchPosition& a, const AMRPatchPosition& b) {
                       return a.nbCells() < b.nbCells();
                     });

    for (Integer p0 = 0; p0 < patch_position.size(); ++p0) {
      AMRPatchPosition& patch_fusion_0 = patch_position[p0];
      if (patch_fusion_0.isNull())
        continue;

      for (Integer p1 = p0 + 1; p1 < patch_position.size(); ++p1) {
        AMRPatchPosition& patch_fusion_1 = patch_position[p1];
        if (patch_fusion_1.isNull())
          continue;

        // if (tm) {
        //   tm->info() << "Check fusion";
        //   tm->info() << " 0 -- Min point : " << patch_fusion_0.minPoint()
        //              << " -- Max point : " << patch_fusion_0.maxPoint()
        //              << " -- Level : " << patch_fusion_0.level()
        //              << " -- NbCells : " << patch_fusion_0.nbCells();
        //   tm->info() << " 1 -- Min point : " << patch_fusion_1.minPoint()
        //              << " -- Max point : " << patch_fusion_1.maxPoint()
        //              << " -- Level : " << patch_fusion_1.level()
        //              << " -- NbCells : " << patch_fusion_1.nbCells();
        // }

        // Si la fusion est possible, le patch 0 est agrandi et le patch 1 devient null.
        if (patch_fusion_0.fusion(patch_fusion_1)) {
          // if (tm)
          //   tm->info() << "Fusion OK";
          fusion = true;
          break;
        }
      }
      if (fusion) {
        break;
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
