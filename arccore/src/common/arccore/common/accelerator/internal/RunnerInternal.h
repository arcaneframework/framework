// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunnerInternal.h                                            (C) 2000-2026 */
/*                                                                           */
/* Internal API of 'Runner' in Arcane.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNNERINTERNAL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNNERINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT RunnerInternal
{
  friend ::Arcane::Accelerator::Runner;
  friend ::Arcane::Accelerator::Impl::RunnerImpl;

 private:

  explicit RunnerInternal(Impl::RunnerImpl* p)
  : m_runner_impl(p)
  {}

 public:

  //! Stops all profiling activities.
  static void stopAllProfiling();

  /*!
   * \brief Finalizes execution.
   *
   * This is used to display certain statistics and release resources.
   */
  static void finalize(ITraceMng* tm);

 public:

  // The following methods that manage profiling act on
  // the runtime (CUDA, ROCM, ...) associated with the runner. For example, if we
  // have two runners associated with CUDA, if we call startProfiling() for one
  // then isProfilingActive() will be true for the second runner.

  //! Indicates whether profiling is active for the associated runtime
  bool isProfilingActive();
  //! Starts profiling for the associated runtime
  void startProfiling();
  //! Stops profiling for the associated runtime
  void stopProfiling();

  /*!
   * \brief Displays profiling information.
   *
   * If it is active, profiling is temporarily stopped and restarted.
   */
  void printProfilingInfos(std::ostream& o);

 private:

  Impl::RunnerImpl* m_runner_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
