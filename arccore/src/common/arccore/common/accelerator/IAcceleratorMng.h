// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAcceleratorMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Accelerator manager interface.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_IACCELERATORMNG_H
#define ARCCORE_COMMON_ACCELERATOR_IACCELERATORMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accelerator manager interface.
 *
 * This interface allows retrieving an instance of Runner and RunQueue
 * associated with a context. You must call initialize() to create these two
 * instances, which can then be retrieved via runner() or queue().
 *
 * It is necessary to call initialize() before accessing
 * methods such as defaultRunner() or defaultQueue().
 */
class ARCCORE_COMMON_EXPORT IAcceleratorMng
{
 public:

  virtual ~IAcceleratorMng() = default;

 public:

  /*!
   * \brief Initializes the instance.
   *
   * \pre isInitialized()==false
   */
  virtual void initialize(const AcceleratorRuntimeInitialisationInfo& runtime_info) =0;

  //! Indicates if the instance has been initialized via the call to initialize()
  virtual bool isInitialized() const =0;

  /*!
   * \brief Default runner.
   *
   * \note This method will eventually be obsolete. It is preferable to use
   * the runner() method instead because it is always valid.
   *
   * The returned pointer remains the property of this instance.
   *
   * \pre isInitialized()==true
   */
  virtual Runner* defaultRunner() =0;

  /*!
   * \brief Default run queue.
   *
   * The returned pointer remains the property of this instance.
   *
   * \note This method will eventually be obsolete. It is preferable to use
   * the queue() method instead because it is always valid.
   *
   * * \pre isInitialized()==true
   */
  virtual RunQueue* defaultQueue() =0;

 public:

  /*!
   * \brief Runner associated with the instance.
   *
   * If the instance has been initialized, returns *defaultRunner().
   * Otherwise, returns a null Runner instance.
   */
  virtual Runner runner() = 0;

  /*!
   * \brief Run queue associated with the instance.
   *
   * If the instance has been initialized, returns *defaultQueue().
   * Otherwise, returns a null queue.
   */
  virtual RunQueue queue() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
