// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionLevelGroup.h                                (C) 2000-2026 */
/*                                                                           */
/* Groupe de position de patch AMR réparti par niveau.                       */
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
 * \brief Classe permettant de stocker les patchs par niveau.
 *
 * Attention : cette classe est utilisé lors de la construction des patchs,
 * un patch de niveau 0 désigne en faite un futur patch de niveau 1.
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
  * \brief Méthode permettant de fusionner tous les patchs d'un certain niveau
  * qui peuvent l'être.
  * \param level Le niveau à fusionner.
  */
  void fusionPatches(Int32 level);

  /*!
  * \brief Méthode permettant de fusionner un maximum de patch du tableau
  * passé en paramètre.
  * \param patch_position [IN/OUT] Le tableau des patchs.
  * \param remove_null Doit-on supprimer les patchs devenus null ?
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
