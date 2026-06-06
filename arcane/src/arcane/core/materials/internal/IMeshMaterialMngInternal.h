// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMngInternal.h                                  (C) 2000-2026 */
/*                                                                           */
/* Internal Arcane API for 'IMeshMaterialMng'.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal Arcane API for 'IMeshMaterialMng'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialMngInternal
{
 public:

  virtual ~IMeshMaterialMngInternal() = default;

 public:

  /*!
   * \brief Adds the variable \a var.
   *
   * This method must not be called directly. References to the variables
   * call it if necessary. This method must be called with the variableLock() active.
   */
  virtual void addVariable(IMeshMaterialVariable* var) = 0;

  /*!
   * \brief Removes the variable \a var.
   *
   * This method must not be called directly. References to the variables
   * call it if necessary. This method must be called with the variableLock()
   * active. Note that this function does not call the delete operator on \a var.
   */
  virtual void removeVariable(IMeshMaterialVariable* var) = 0;

  /*!
   * \brief Modifier implementation.
   *
   * This modifier allows changing the list of meshes composing a medium
   * or a material. This method should in principle not be called directly.
   * To modify, it is better to use an instance of MeshMaterialModifier
   * which guarantees that the update functions are called.
   */
  virtual MeshMaterialModifierImpl* modifier() = 0;

  /*!
   * \brief List of information to index material variables.
   */
  virtual ConstArrayView<MeshMaterialVariableIndexer*> variablesIndexer() = 0;

  /*!
   * \brief Synchronizer for material and medium variables across all meshes.
   */
  virtual IMeshMaterialVariableSynchronizer* allCellsMatEnvSynchronizer() = 0;

  /*!
   * \brief Synchronizer for medium-only variables across all meshes.
   */
  virtual IMeshMaterialVariableSynchronizer* allCellsEnvOnlySynchronizer() = 0;

 public:

  /*!
   * \brief Returns the "connectivity" table CellLocalId -> AllEnvCell
   * intended to be used in a RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * in conjunction with the ENUMERATE_CELL_ALLENVCELL macro
   */
  virtual AllCellToAllEnvCellContainer* getAllCellToAllEnvCellContainer() const = 0;

  /*!
   * \brief Constructs the "connectivity" table CellLocalId -> AllEnvCell
   * intended to be used in a RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * in conjunction with the ENUMERATE_CELL_ALLENVCELL macro
   *
   * If no allocator is specified, the method
   * platform::getDefaultDataAllocator() is used
   */
  virtual void createAllCellToAllEnvCell() = 0;

  /*!
   * \brief ComponentItemSharedInfo instance for a constituent
   *
   * The value of \a level must be LEVEL_MATERIAL or LEVEL_ENVIRONMENT
   */
  virtual ComponentItemSharedInfo* componentItemSharedInfo(Int32 level) const = 0;

  //! Default run queue.
  virtual RunQueue& runQueue() const = 0;

  //! List of asynchronous queues
  virtual Accelerator::RunQueuePool& asyncRunQueuePool() const = 0;

  //! Ratio for additional capacity to allocate when resizing variables.
  virtual Real additionalCapacityRatio() const = 0;

  //! Indicates whether the accelerator API is used to position the values of ConstituentItemVectorImpl
  virtual bool isUseAcceleratorForConstituentItemVector() const = 0;

  /*!
   * \brief Run queue for the \a policy.
   *
   * If \a policy equals eExecutionPolicy::None, then runQueue() is returned.
   * The other possible values are eExecutionPolicy::Sequential
   * or eExecutionPolicy::Thread.
   */
  virtual RunQueue runQueue(Accelerator::eExecutionPolicy policy) const = 0;

  /*!
   * \brief View of the array corresponding to a selection across all entities.
   *
   * Returns a view of an array \a v sized to the number of meshes in the mesh
   * and having `v[i] == i` for all \a i. This array is invalidated if the
   * number of meshes changes. It is used notably for indexed selections
   * (via the ConstituentItemIndexedSelectionView class).
   */
  virtual SmallSpan<const Int32> identitySelectionView() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
