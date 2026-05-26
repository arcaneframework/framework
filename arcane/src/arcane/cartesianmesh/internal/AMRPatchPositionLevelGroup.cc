// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.cc                               (C) 2000-2026 */
/*                                                                           */
/* AMR patch position group distributed by level.                            */
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
  // Fusion algorithm.
  // First, we sort the patches from the smallest number of cells to the largest number of cells (optional).
  // Then, for each patch, we check if it can be merged with another.
  // If a fusion is achieved, we restart the algorithm until no more fusions can be performed.
  bool fusion = true;
  while (fusion) {
    fusion = false;

    // The sort allows merging the smallest patches first, which helps to balance (a little) the patch sizes.
    // Warning: removing the sort changes the test hashes.
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

        // If fusion is possible, patch 0 is enlarged and patch 1 becomes null.
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
