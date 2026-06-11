// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
   * \brief Method allowing cell refinement using the "II_Refine" flag.
   */
  virtual void refine() = 0;

  /*!
   * \brief Method allowing coarsening of level 0 cells.
   *
   * A cell level -1 will be created with parent cells to the level 0 cells,
   * and then all levels will be incremented by 1. The level created
   * by this method will therefore be the new level 0.
   */
  virtual void createSubLevel() = 0;

  /*!
   * \brief Method allowing removal of cells marked with the "II_Coarsen" flag.
   *
   * The owners of faces and nodes having marked cells and unmarked cells
   * are likely to be updated.
   *
   * \param update_parent_flag If true, the flags of the parents will be
   * updated. This includes activating the parent cells.
   */
  virtual void coarsen(bool update_parent_flag) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
