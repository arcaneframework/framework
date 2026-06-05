// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellContainer.h                              (C) 2000-2024 */
/*                                                                           */
/* Data container for 'AllCellToAllEnvCell'.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_ALLCELLTOALLENVCELLCONTAINER_H
#define ARCANE_MATERIALS_INTERNAL_ALLCELLTOALLENVCELLCONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshMaterialAcceleratorUnitTest;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Connectivity table of 'Cell' to its 'AllEnvCell' intended
 *        for use on accelerator.
 *
 * Class that maintains the connectivity of all meshes
 * \a Cell to all their meshes \a AllEnvCell.
 *
 * An instance is created via the create() method.
 *
 * The initialization cost is high; memory must be allocated and structures filled. We iterate through all meshes and for each mesh we call
 * the CellToAllEnvCellConverter.
 *
 * Once the instance is created, it must be updated every time that
 * the topology of materials/environments changes (which is also expensive).
 *
 * This class is an internal class and should not be manipulated directly.
 * One must use the associated helpers in the IMeshMaterialMng and
 * the CellToAllEnvCellAccessor class.
 */
class ARCANE_MATERIALS_EXPORT AllCellToAllEnvCellContainer
{
 public:

  class Impl;

 public:

  explicit AllCellToAllEnvCellContainer(IMeshMaterialMng* mm);
  ~AllCellToAllEnvCellContainer();

 public:

  //! Copies forbidden
  AllCellToAllEnvCellContainer(const AllCellToAllEnvCellContainer&) = delete;
  AllCellToAllEnvCellContainer& operator=(const AllCellToAllEnvCellContainer&) = delete;
  AllCellToAllEnvCellContainer& operator=(AllCellToAllEnvCellContainer&&) = delete;

 public:

  /*!
   * \brief Alternative creation function. It is necessary to wait until the data
   * related to the materials is finalized.
   *
   * The difference lies in memory management.
   * Here, a compromise is applied to the size of the cid -> envcells table
   * where the size of the array for storing the envcells of a cell is equal to the size
   * max of the number of environments present at time t in a mesh.
   * This allows avoiding memory allocations in the internal loop and
   * in a systematic way.
   * => Performance gain to be evaluated.
   */
  void initialize();

  /*!
   * \brief Method to provide the maximum number of environments
   * present on a mesh at time t.
   *
   * Performing this operation at a given moment allows
   * having a max value <= the total number of environments present
   * in the JDD (and thus saving some memory).
   */
  Int32 computeMaxNbEnvPerCell() const;

  /*!
   * We check if the max number of envs per cell at time t has changed,
   * and if so, we force the reconstruction of the table.
   * Is called by the forceRecompute of the IMeshMaterialMng
   */
  void bruteForceUpdate();

  void reset();

  AllCellToAllEnvCell view() const;

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
