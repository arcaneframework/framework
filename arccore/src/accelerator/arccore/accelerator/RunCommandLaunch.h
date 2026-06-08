// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunch.h                                          (C) 2000-2026 */
/*                                                                           */
/* RunCommand for hierarchical parallelism.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCH_H
#define ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/RunCommandLaunchImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const Impl::ExtendedLaunchLoop<LoopBoundType, RemainingArgs...>& ex_loop)
-> Impl::ExtendedLaunchRunCommand<LoopBoundType, RemainingArgs...>
{
  return { command, ex_loop.m_bounds, ex_loop.m_remaining_args };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// For Sycl, the iterator type cannot be the same on the host and
// the device because a 'sycl::nd_item' is required and it is not possible to
// construct one (no default constructor). We therefore use
// a template lambda and the iterator type is a template parameter

/*!
 * \brief Macro to launch a command using hierarchical,
 * possibly cooperative, parallelism.
 *
 * \a bounds must be an instance of type Arcane::Accelerator::WorkGroupLoopRange
 * or Arcane::Accelerator::CooperativeWorkGroupLoopRange.
 *
 * The creation of these instances is done by calling
 * Arcane::Accelerator::makeWorkGroupLoopRange() or
 * Arcane::Accelerator::makeCooperativeWorkGroupLoopRange().
 *
 * \a iter_name will be of type Arcane::Accelerator::WorkGroupLoopContext or
 * Arcane::Accelerator::CooperativeWorkGroupLoopContext (except
 * for the execution policy Arcane::Accelerator::eExecutionPolicy::SYCL
 * where the type is templated and is different if running on the host or
 * on the accelerator).
 */
#if defined(ARCCORE_COMPILING_SYCL)
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::Impl::makeLaunch(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(auto iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))
#else
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::Impl::makeLaunch(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
