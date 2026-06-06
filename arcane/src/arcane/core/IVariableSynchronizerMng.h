// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMng.h                                  (C) 2000-2023 */
/*                                                                           */
/* Interface of the variable synchronization manager.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
#define ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableSynchronizerMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the variable synchronization manager.
 */
class ARCANE_CORE_EXPORT IVariableSynchronizerMng
{
 public:

  virtual ~IVariableSynchronizerMng() = default;

 public:

  //! Associated parallelism manager
  virtual IParallelMng* parallelMng() const = 0;

  /*!
   * \brief Event sent at the beginning and end of synchronization.
   *
   * This event is sent when calling the methods
   * IVariableSynchronizer::synchronize(IVariable* var)
   * and IVariableSynchronizer::synchronize(VariableCollection vars) for all
   * instances of IVariableSynchronizer.
   */
  virtual EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() = 0;

  /*!
   * \brief Sets the comparison level between values before and after synchronization.
   *
   * The level must be the same across all ranks of parallelMng().
   */
  virtual void setSynchronizationCompareLevel(Int32 v) = 0;

  //! Comparison level of values before and after synchronization
  virtual Int32 synchronizationCompareLevel() const = 0;

  //! Indicates whether comparisons of values before and after synchronization are performed.
  virtual bool isSynchronizationComparisonEnabled() const = 0;

  /*!
   * \brief Prints statistics to the stream \a ostr.
   *
   * Statistics must be processed via the call to flushPendingStats()
   * before calling this method.
   */
  virtual void dumpStats(std::ostream& ostr) const = 0;

  /*!
   * \brief Processes pending statistics.
   *
   * This method does nothing if isComparisonEnabled() is \a false.
   *
   * This method is collective on parallelMng().
   */
  virtual void flushPendingStats() = 0;

 public:

  virtual IVariableSynchronizerMngInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
