// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLoop.h                                            (C) 2000-2025 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une commande.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour compatibilité avec l'existant
#include "arcane/utils/ConcurrencyUtils.h"

#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/KernelLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique la lambda \a func sur une boucle \a bounds.
 *
 * La lambda \a func est appliqué à la commande \a command.
 * \a bound est le type de la boucle. Les types supportés sont:
 *
 * - SimpleForLoopRanges
 * - ComplexForLoopRanges
 *
 * Les arguments supplémentaires \a other_args sont utilisés pour supporter
 * des fonctionnalités telles que les réductions (ReducerSum2, ReducerMax2, ...)
 * ou la gestion de la mémoire locale (via RunCommandLocalMemory).
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
_applyGenericLoop(RunCommand& command, LoopBoundType bounds,
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
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(impl::doDirectGPULambdaArrayBounds2) < LoopBoundType, Lambda, RemainingArgs... >, func, bounds, other_args...);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(impl::doDirectGPULambdaArrayBounds2) < LoopBoundType, Lambda, RemainingArgs... >, func, bounds, other_args...);
    break;
  case eExecutionPolicy::SYCL:
    _applyKernelSYCL(launch_info, ARCANE_KERNEL_SYCL_FUNC(impl::DoDirectSYCLLambdaArrayBounds) < LoopBoundType, Lambda, RemainingArgs... > {}, func, bounds, other_args...);
    break;
  case eExecutionPolicy::Sequential:
    arcaneSequentialFor(bounds, func, other_args...);
    break;
  case eExecutionPolicy::Thread:
    arccoreParallelFor(bounds, launch_info.loopRunInfo(), func, other_args...);
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour conserver les arguments d'une RunCommand.
 *
 * `LoopBoundType` est le type de la boucle. Par exemple, ce peut être
 * une SimpleForLoopRanges ou une ComplexForLoopRanges.
 */
template <typename LoopBoundType, typename... RemainingArgs>
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour gérer les paramètres supplémentaires des commandes.
 */
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

template <typename LoopBoundType, typename... RemainingArgs> auto
makeExtendedArrayBoundLoop(const LoopBoundType& bounds, RemainingArgs... args)
-> ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>
{
  return ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>(bounds, args...);
}

template <typename LoopBoundType, typename... RemainingArgs> auto
makeExtendedLoop(const LoopBoundType& bounds, RemainingArgs... args)
-> ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>
{
  return ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>(bounds, args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds.
 *
 * \a other_args contient les éventuels arguments supplémentaires passés à
 * la lambda.
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
runExtended(RunCommand& command, LoopBoundType bounds,
            const Lambda& func, const std::tuple<RemainingArgs...>& other_args)
{
  std::apply([&](auto... vs) { _applyGenericLoop(command, bounds, func, vs...); }, other_args);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ExtentType> auto
operator<<(RunCommand& command, const ArrayBounds<ExtentType>& bounds)
  -> impl::ArrayBoundRunCommand<SimpleForLoopRanges<ExtentType::rank(), Int32>>
{
  return { command, bounds };
}

template <typename LoopBoundType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const impl::ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>& ex_loop)
  -> impl::ArrayBoundRunCommand<LoopBoundType, RemainingArgs...>
{
  return { command, ex_loop.m_bounds, ex_loop.m_remaining_args };
}

template <int N> impl::ArrayBoundRunCommand<SimpleForLoopRanges<N>>
operator<<(RunCommand& command, const SimpleForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

template <int N> impl::ArrayBoundRunCommand<ComplexForLoopRanges<N>>
operator<<(RunCommand& command, const ComplexForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
inline void operator<<(ArrayBoundRunCommand<LoopBoundType, RemainingArgs...>&& nr, const Lambda& f)
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
#define RUNCOMMAND_LOOP(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOPN(iter_name, N, ...) \
  A_FUNCINFO << Arcane::ArrayBounds<typename Arcane::MDDimType<N>::DimType>(__VA_ARGS__) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<N> iter_name)

//! Boucle 2D sur accélérateur
#define RUNCOMMAND_LOOP2(iter_name, x1, x2) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim2>(x1, x2) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<2> iter_name)

//! Boucle 3D sur accélérateur
#define RUNCOMMAND_LOOP3(iter_name, x1, x2, x3) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim3>(x1, x2, x3) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<3> iter_name)

//! Boucle 4D sur accélérateur
#define RUNCOMMAND_LOOP4(iter_name, x1, x2, x3, x4) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim4>(x1, x2, x3, x4) << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<4> iter_name)

/*!
 * \brief Boucle 1D sur accélérateur avec arguments supplémentaires.
 *
 * Cette macro permet d'ajouter des arguments. Ces arguments peuvent être
 * des valeurs à réduire (telles que les classes Arcane::Accelerator::ReducerSum2,
 * Arcane::Accelerator::ReducerMax2 ou Arcane::Accelerator::ReducerMin2) ou des données
 * en mémoire locale (via la classe Arcane::Accelerator::RunCommandLocalMemory).
 */
#define RUNCOMMAND_LOOP1(iter_name, x1, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedArrayBoundLoop(::Arcane::SimpleForLoopRanges<1>(x1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<1> iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

/*!
 * \brief Boucle sur accélérateur pour exécution avec un seul thread.
 */
#define RUNCOMMAND_SINGLE(...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedArrayBoundLoop(::Arcane::SimpleForLoopRanges<1>(1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<1> __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
