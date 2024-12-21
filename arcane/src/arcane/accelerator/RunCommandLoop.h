﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLoop.h                                            (C) 2000-2024 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une commande.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunQueueInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique la lambda \a func sur une boucle \a bounds.
 *
 * \a N est la dimension de la boucle (actuellement 1, 2, 3 ou 4). La lambda
 * \a func est appliqué à la commande \a command. Les arguments supplémentaires
 * sont des fonctor supplémentaires (comme les réductions).
 */
template <int N, template <int T, typename> class LoopBoundType, typename Lambda, typename... RemainingArgs> void
_applyGenericLoop(RunCommand& command, LoopBoundType<N, Int32> bounds,
                  const Lambda& func, const RemainingArgs&... other_args)
{
  Int64 vsize = bounds.nbElement();
  if (vsize == 0)
    return;
  impl::RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(impl::doDirectGPULambdaArrayBounds2) < LoopBoundType<N, Int32>, Lambda, RemainingArgs... >, func, bounds, other_args...);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(impl::doDirectGPULambdaArrayBounds2) < LoopBoundType<N, Int32>, Lambda, RemainingArgs... >, func, bounds, other_args...);
    break;
  case eExecutionPolicy::SYCL:
    _applyKernelSYCL(launch_info, ARCANE_KERNEL_SYCL_FUNC(impl::DoDirectSYCLLambdaArrayBounds) < LoopBoundType<N, Int32>, Lambda, RemainingArgs... > {}, func, bounds, other_args...);
    break;
  case eExecutionPolicy::Sequential:
    arcaneSequentialFor(bounds, func, other_args...);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelFor(bounds, launch_info.computeParallelLoopOptions(), func, other_args...);
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename... RemainingArgs>
class ExtendedArrayBoundLoop
{
 public:

  ExtendedArrayBoundLoop(const LoopBoundType& bounds, RemainingArgs... args)
  : m_bounds(bounds)
  , m_remaining_args(args...)
  {
  }
  LoopBoundType m_bounds;
  std::tuple<RemainingArgs...> m_remaining_args;
};

template <typename ExtentType, typename... RemainingArgs> auto
makeExtendedArrayBoundLoop(const ArrayBounds<ExtentType>& bounds, RemainingArgs... args)
-> ExtendedArrayBoundLoop<ArrayBounds<ExtentType>, RemainingArgs...>
{
  return ExtendedArrayBoundLoop<ArrayBounds<ExtentType>, RemainingArgs...>(bounds, args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <typename ExtentType, typename Lambda> void
run(RunCommand& command, ArrayBounds<ExtentType> bounds, const Lambda& func)
{
  impl::_applyGenericLoop(command, SimpleForLoopRanges<ExtentType::rank()>(bounds), func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <int N, typename Lambda> void
run(RunCommand& command, SimpleForLoopRanges<N, Int32> bounds, const Lambda& func)
{
  impl::_applyGenericLoop(command, bounds, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <int N, typename Lambda> void
run(RunCommand& command, ComplexForLoopRanges<N, Int32> bounds, const Lambda& func)
{
  impl::_applyGenericLoop(command, bounds, func);
}

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <int N, template <int T, typename> class LoopBoundType, typename Lambda, typename... RemainingArgs> void
runExtended(RunCommand& command, LoopBoundType<N, Int32> bounds,
            const Lambda& func, const std::tuple<RemainingArgs...>& other_args)
{
  std::apply([&](auto... vs) { impl::_applyGenericLoop(command, bounds, func, vs...); }, other_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int N, typename LoopBoundType, typename... RemainingArgs>
class ArrayBoundRunCommand
{
 public:

  ArrayBoundRunCommand(RunCommand& command, const LoopBoundType& bounds)
  : m_command(command)
  , m_bounds(bounds)
  {
  }
  ArrayBoundRunCommand(RunCommand& command, const LoopBoundType& bounds, const std::tuple<RemainingArgs...>& args)
  : m_command(command)
  , m_bounds(bounds)
  , m_remaining_args(args)
  {
  }
  RunCommand& m_command;
  LoopBoundType m_bounds;
  std::tuple<RemainingArgs...> m_remaining_args;
};

template <typename ExtentType> auto
operator<<(RunCommand& command, const ArrayBounds<ExtentType>& bounds)
-> ArrayBoundRunCommand<ExtentType::rank(), SimpleForLoopRanges<ExtentType::rank(), Int32>>
{
  return { command, bounds };
}

template <typename ExtentType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const impl::ExtendedArrayBoundLoop<ExtentType, RemainingArgs...>& ex_loop)
-> ArrayBoundRunCommand<1, SimpleForLoopRanges<1, Int32>, RemainingArgs...>
{
  return { command, ex_loop.m_bounds, ex_loop.m_remaining_args };
}

template <int N> ArrayBoundRunCommand<N, SimpleForLoopRanges<N>>
operator<<(RunCommand& command, const SimpleForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

template <int N> ArrayBoundRunCommand<N, ComplexForLoopRanges<N>>
operator<<(RunCommand& command, const ComplexForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

template <int N, template <int, typename> class ForLoopBoundType, typename Lambda, typename... RemainingArgs>
void operator<<(ArrayBoundRunCommand<N, ForLoopBoundType<N, Int32>, RemainingArgs...>&& nr, const Lambda& f)
{
  if constexpr (sizeof...(RemainingArgs) > 0) {
    runExtended(nr.m_command, nr.m_bounds, f, nr.m_remaining_args);
  }
  else {
    run(nr.m_command, nr.m_bounds, f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP(iter_name, bounds) \
  A_FUNCINFO << bounds << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::IndexType iter_name)

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOPN(iter_name, N, ...) \
  A_FUNCINFO << Arcane::ArrayBounds<typename Arcane::MDDimType<N>::DimType>(__VA_ARGS__) << [=] ARCCORE_HOST_DEVICE(Arcane::ArrayIndex<N> iter_name)

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP2(iter_name, x1, x2) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim2>(x1, x2) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<2> iter_name)

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP3(iter_name, x1, x2, x3) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim3>(x1, x2, x3) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<3> iter_name)

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP4(iter_name, x1, x2, x3, x4) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim4>(x1, x2, x3, x4) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<4> iter_name)

/*!
 * \brief Boucle sur accélérateur avec arguments supplémentaires pour les réductions.
 *
 * Cette macro permet d'ajouter des arguments
 * pour chaque valeur à réduire. Les arguments doivent être des instances des
 * classes Arcane::Accelerator::ReducerSum2, Arcane::Accelerator::ReducerMax2 ou Arcane::Accelerator::ReducerMin2.
 */
#define RUNCOMMAND_LOOP1(iter_name, x1, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedArrayBoundLoop(Arcane::ArrayBounds<MDDim1>(x1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::ArrayIndex<1> iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

/*!
 * \brief Boucle sur accélérateur pour exécution avec un seul thread.
 */
#define RUNCOMMAND_SINGLE(...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedArrayBoundLoop(Arcane::ArrayBounds<MDDim1>(1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::ArrayIndex<1> __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

/*!
 * \brief Boucle sur accélérateur.
 *
 * \deprecated Utiliser RUNCOMMAND_LOOP1() à la place.
 */
#define RUNCOMMAND_LOOP1_EX(iter_name, x1, ...) \
  RUNCOMMAND_LOOP1(iter_name, x1, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
