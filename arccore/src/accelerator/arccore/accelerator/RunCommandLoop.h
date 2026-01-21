// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLoop.h                                            (C) 2000-2026 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une commande.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_RUNCOMMANDLOOP_H
#define ARCCORE_ACCELERATOR_RUNCOMMANDLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/SequentialFor.h"
#include "arccore/common/StridedLoopRanges.h"
#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/concurrency/ParallelFor.h"
#include "arccore/accelerator/KernelLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

template <int N, typename IndexType_>
constexpr ARCCORE_HOST_DEVICE SimpleForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexCudaHip(const SimpleForLoopRanges<N, IndexType_>& bounds, Int32 i)
{
  return bounds.getIndices(i);
}

template <int N, typename IndexType_>
constexpr ARCCORE_HOST_DEVICE ComplexForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexCudaHip(const ComplexForLoopRanges<N, IndexType_>& bounds, Int32 i)
{
  return bounds.getIndices(i);
}

#if defined(ARCCORE_COMPILING_SYCL)

template <int N, typename IndexType_>
SimpleForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexSycl(const SimpleForLoopRanges<N, IndexType_>& bounds, sycl::nd_item<1> x)
{
  return bounds.getIndices(static_cast<Int32>(x.get_global_id(0)));
}

template <int N, typename IndexType_>
ComplexForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexSycl(const ComplexForLoopRanges<N, IndexType_>& bounds, sycl::nd_item<1> x)
{
  return bounds.getIndices(static_cast<Int32>(x.get_global_id(0)));
}

template <int N, typename IndexType_>
SimpleForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexSycl(const SimpleForLoopRanges<N, IndexType_>& bounds, sycl::id<1> x)
{
  return bounds.getIndices(static_cast<Int32>(x));
}

template <int N, typename IndexType_>
ComplexForLoopRanges<N, IndexType_>::LoopIndexType
arcaneGetLoopIndexSycl(const ComplexForLoopRanges<N, IndexType_>& bounds, sycl::id<1> x)
{
  return bounds.getIndices(static_cast<Int32>(x));
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// On utilise 'Argument dependent lookup' pour trouver 'arcaneGetLoopIndexCudaHip'
#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> __global__ void
doDirectGPULambdaArrayBounds2(LoopBoundType bounds, Lambda func, RemainingArgs... remaining_args)
{
  using namespace Arcane::Accelerator::Impl;

  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  if constexpr (requires { bounds.nbStride(); }) {
    // Test expérimental pour utiliser un pas de la taille
    // de la grille. Le nombre de pas est donné par bounds.nbStride().
    Int32 nb_grid_stride = bounds.nbStride();
    Int32 offset = blockDim.x * gridDim.x;
    for (Int32 k = 0; k < nb_grid_stride; ++k) {
      Int32 true_i = i + (offset * k);
      if (true_i < bounds.nbOriginalElement()) {
        body(arcaneGetLoopIndexCudaHip(bounds, true_i), remaining_args...);
      }
    }
  }
  else {
    if (i < bounds.nbElement()) {
      body(arcaneGetLoopIndexCudaHip(bounds, i), remaining_args...);
    }
  }
  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

//! Boucle N-dimension sans indirection
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
class DoDirectSYCLLambdaArrayBounds
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shared_memory,
                  LoopBoundType bounds, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();
    Int32 i = static_cast<Int32>(x.get_global_id(0));
    SyclKernelRemainingArgsHelper::applyAtBegin(x, shared_memory, remaining_args...);
    if (i < bounds.nbElement()) {
      // Si possible, on passe \a x en argument
      body(arcaneGetLoopIndexSycl(bounds, x), remaining_args...);
    }
    SyclKernelRemainingArgsHelper::applyAtEnd(x, shared_memory, remaining_args...);
  }
  void operator()(sycl::id<1> x, LoopBoundType bounds, Lambda func) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x);
    if (i < bounds.nbElement()) {
      body(arcaneGetLoopIndexSycl(bounds, i));
    }
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
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
 * ou la gestion de la mémoire locale (via LocalMemory).
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
_applyGenericLoop(RunCommand& command, LoopBoundType bounds,
                  const Lambda& func, const RemainingArgs&... other_args)
{
  Int64 vsize = bounds.nbElement();
  if (vsize == 0)
    return;
#if defined(ARCCORE_EXPERIMENTAL_GRID_STRIDE) && defined(ARCCORE_COMPILING_CUDA_OR_HIP)
  using TrueLoopBoundType = Impl::StridedLoopRanges<LoopBoundType>;
  TrueLoopBoundType bounds2(command.nbStride(), bounds);
  Impl::RunCommandLaunchInfo launch_info(command, bounds2.strideValue());
#else
  using TrueLoopBoundType = LoopBoundType;
  [[maybe_unused]] const TrueLoopBoundType& bounds2 = bounds;
  Impl::RunCommandLaunchInfo launch_info(command, vsize);
#endif
  launch_info.beginExecute();
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    ARCCORE_KERNEL_CUDA_FUNC((Impl::doDirectGPULambdaArrayBounds2<TrueLoopBoundType, Lambda, RemainingArgs...>),
                             launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::HIP:
    ARCCORE_KERNEL_HIP_FUNC((Impl::doDirectGPULambdaArrayBounds2<TrueLoopBoundType, Lambda, RemainingArgs...>),
                            launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::SYCL:
    ARCCORE_KERNEL_SYCL_FUNC((Impl::DoDirectSYCLLambdaArrayBounds<LoopBoundType, Lambda, RemainingArgs...>{}),
                             launch_info, func, bounds, other_args...);
    break;
  case eExecutionPolicy::Sequential:
    arccoreSequentialFor(bounds, func, other_args...);
    break;
  case eExecutionPolicy::Thread:
    arccoreParallelFor(bounds, launch_info.loopRunInfo(), func, other_args...);
    break;
  default:
    ARCCORE_FATAL("Invalid execution policy '{0}'", exec_policy);
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

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <typename LoopBoundType, typename Lambda> void
runGeneric(RunCommand& command, const LoopBoundType& bounds, const Lambda& func)
{
  Impl::_applyGenericLoop(command, bounds, func);
}

// Spécialisation pour ArrayBound.
//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <typename ExtentType, typename Lambda> void
runGeneric(RunCommand& command, ArrayBounds<ExtentType> bounds, const Lambda& func)
{
  Impl::_applyGenericLoop(command, SimpleForLoopRanges<ExtentType::rank()>(bounds), func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: a rendre obsolète et supprimer
//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <typename ExtentType, typename Lambda> void
run(RunCommand& command, ArrayBounds<ExtentType> bounds, const Lambda& func)
{
  Impl::_applyGenericLoop(command, SimpleForLoopRanges<ExtentType::rank()>(bounds), func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: a rendre obsolète et supprimer
//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <int N, typename Lambda> void
run(RunCommand& command, SimpleForLoopRanges<N, Int32> bounds, const Lambda& func)
{
  Impl::_applyGenericLoop(command, bounds, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: a rendre obsolète et supprimer
//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template <int N, typename Lambda> void
run(RunCommand& command, ComplexForLoopRanges<N, Int32> bounds, const Lambda& func)
{
  Impl::_applyGenericLoop(command, bounds, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ExtentType> auto
operator<<(RunCommand& command, const ArrayBounds<ExtentType>& bounds)
-> Impl::ArrayBoundRunCommand<SimpleForLoopRanges<ExtentType::rank(), Int32>>
{
  return { command, bounds };
}

template <typename LoopBoundType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const Impl::ExtendedArrayBoundLoop<LoopBoundType, RemainingArgs...>& ex_loop)
-> Impl::ArrayBoundRunCommand<LoopBoundType, RemainingArgs...>
{
  return { command, ex_loop.m_bounds, ex_loop.m_remaining_args };
}

template <int N> Impl::ArrayBoundRunCommand<SimpleForLoopRanges<N>>
operator<<(RunCommand& command, const SimpleForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

template <int N> Impl::ArrayBoundRunCommand<ComplexForLoopRanges<N>>
operator<<(RunCommand& command, const ComplexForLoopRanges<N, Int32>& bounds)
{
  return { command, bounds };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
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
    runGeneric(nr.m_command, nr.m_bounds, f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::Impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))

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
 * en mémoire locale (via la classe Arcane::Accelerator::LocalMemory).
 */
#define RUNCOMMAND_LOOP1(iter_name, x1, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::Impl::makeExtendedArrayBoundLoop(::Arcane::SimpleForLoopRanges<1>(x1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<1> iter_name __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))

/*!
 * \brief Boucle sur accélérateur pour exécution avec un seul thread.
 */
#define RUNCOMMAND_SINGLE(...) \
  A_FUNCINFO << ::Arcane::Accelerator::Impl::makeExtendedArrayBoundLoop(::Arcane::SimpleForLoopRanges<1>(1) __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::MDIndex<1> __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
