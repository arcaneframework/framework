// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshAMRPatchMng.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface for managing AMR by patch of a Cartesian mesh.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshAMRPatchMng
{
 public:

  virtual ~ICartesianMeshAMRPatchMng() = default;

 public:

  /*!
   * \brief Method allowing mesh refinement using the "II_Refine" flag.
   */
  virtual void refine() = 0;

  /*!
   * \brief Method allowing coarsening of level 0 meshes.
   *
   * A mesh level -1 will be created with parent meshes to the level 0 meshes,
   * and then all levels will be incremented by 1. The level created
   * by this method will therefore be the new level 0.
   */
  virtual void createSubLevel() = 0;

  /*!
   * \brief Method allowing removal of meshes marked with the "II_Coarsen" flag.
   *
   * The owners of faces and nodes having marked meshes and unmarked meshes
   * are likely to be updated.
   *
   * \param update_parent_flag If true, the flags of the parents will be
   * updated. This includes activating the parent meshes.
   */
  virtual void coarsen(bool update_parent_flag) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H
