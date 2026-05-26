// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.h                                (C) 2000-2026 */
/*                                                                           */
/* Group of AMR patch positions distributed by level.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONLEVELGROUP_H
#define ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONLEVELGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing storage of patches by level.
 *
 * Note: this class is used during patch construction,
 * a level 0 patch actually designates a future level 1 patch.
 */
class AMRPatchPositionLevelGroup
{
 public:

  explicit AMRPatchPositionLevelGroup(Int32 max_level);
  ~AMRPatchPositionLevelGroup();

 public:

  Int32 maxLevel() const;
  ConstArrayView<AMRPatchPosition> patches(Int32 level);
  void addPatch(const AMRPatchPosition& patch);

  /*!
  * \brief Method allowing the merging of all patches of a certain level
  * that can be merged.
  * \param level The level to merge.
  */
  void fusionPatches(Int32 level);

  /*!
  * \brief Method allowing the merging of a maximum number of patches from the array
  * passed as a parameter.
  * \param patch_position [IN/OUT] The array of patches.
  * \param remove_null Should null patches be removed?
  */
  static void fusionPatches(UniqueArray<AMRPatchPosition>& patch_position, bool remove_null);

 private:

  Int32 m_max_level;
  UniqueArray<UniqueArray<AMRPatchPosition>> m_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
