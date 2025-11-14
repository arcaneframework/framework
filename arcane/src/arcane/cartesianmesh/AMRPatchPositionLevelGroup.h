// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.h                                        (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRPATCHPOSITIONLEVELGROUP_H
#define ARCANE_CARTESIANMESH_AMRPATCHPOSITIONLEVELGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPosition.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/utils/Vector3.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMRPatchPositionLevelGroup
{
 public:
  AMRPatchPositionLevelGroup(Integer max_level);
  ~AMRPatchPositionLevelGroup();

 public:

  Integer maxLevel();
  ConstArrayView<AMRPatchPosition> patches(Integer level);
  void addPatch(AMRPatchPosition patch);
  void fusionPatches(Integer level);
  static void fusionPatches(UniqueArray<AMRPatchPosition>& patch_position, bool remove_null);

 private:
  Integer m_max_level;
  UniqueArray<UniqueArray<AMRPatchPosition>> m_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

