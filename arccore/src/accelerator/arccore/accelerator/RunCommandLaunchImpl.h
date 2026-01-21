// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunchImpl.h                                      (C) 2000-2026 */
/*                                                                           */
/* Implémentation d'une RunCommand pour le parallélisme hiérarchique.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
#define ARCCORE_ACCELERATOR_RUNCOMMANDLAUNCHIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/WorkGroupLoopRange.h"

#include "arccore/common/SequentialFor.h"
#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/concurrency/ParallelFor.h"
#include "arccore/accelerator/KernelLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Classe pour exécuter en séquentiel sur l'hôte une partie de la boucle.
 */
class WorkGroupSequentialForHelper
{
 public:

  //! Applique le fonctor \a func sur une boucle séqentielle.
  template <typename Lambda, typename... RemainingArgs> static void
  apply(Int32 begin_index, Int32 nb_loop, WorkGroupLoopRange bounds,
        const Lambda& func, RemainingArgs... remaining_args)
  {
    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
    const Int32 group_size = bounds.groupSize();
    Int32 loop_index = begin_index * group_size;
    for (Int32 i = begin_index; i < (begin_index + nb_loop); ++i) {
      // Pour la dernière itération de la boucle, le nombre d'éléments actifs peut-être
      // inférieur à la taille d'un groupe si \a total_nb_element n'est pas
      // un multiple de \a group_size.
      Int32 nb_active = bounds.nbActiveItem(i);
      func(WorkGroupLoopContext(loop_index, i, group_size, nb_active), remaining_args...);
      loop_index += group_size;
    }

    ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// On utilise 'Argument dependent lookup' pour trouver 'arcaneGetLoopIndexCudaHip'
#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> __global__ void
doHierarchicalLaunchCudaHip(LoopBoundType bounds, Lambda func, RemainingArgs... remaining_args)
{
  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  //auto privatizer = privatize(func);
  //auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  if (i < bounds.nbElement()) {
    // NOTE: les arguments bounds et i ne sont pas utilisés dans arcaneGetLoopIndexCudaHip()
    func(arcaneGetLoopIndexCudaHip(bounds, i), remaining_args...);
  }
  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

#endif

#if defined(ARCCORE_COMPILING_SYCL)

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
class doHierarchicalLaunchSycl
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shared_memory,
                  LoopBoundType bounds, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    Int32 i = static_cast<Int32>(x.get_global_id(0));
    SyclKernelRemainingArgsHelper::applyAtBegin(x, shared_memory, remaining_args...);
    if (i < bounds.nbElement()) {
      // Si possible, on passe \a x en argument
      func(arcaneGetLoopIndexSycl(bounds, x), remaining_args...);
    }
    SyclKernelRemainingArgsHelper::applyAtEnd(x, shared_memory, remaining_args...);
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique la lambda \a func sur une boucle \a bounds.
 *
 * La lambda \a func est appliqué à la commande \a command.
 * \a bound est le type de la boucle. Les types supportés sont:
 *
 * - WorkGroupLoopRange
 * - CooperativeWorkGroupLoopRange
 *
 * Les arguments supplémentaires \a other_args sont utilisés pour supporter
 * des fonctionnalités telles que les réductions (ReducerSum2, ReducerMax2, ...)
 * ou la gestion de la mémoire locale (via LocalMemory).
 */
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> void
_doHierarchicalLaunch(RunCommand& command, LoopBoundType bounds,
                      const Lambda& func, const RemainingArgs&... other_args)
{
  Int64 vsize = bounds.nbElement();
  if (vsize == 0)
    return;
  using TrueLoopBoundType = LoopBoundType;
  [[maybe_unused]] const TrueLoopBoundType& bounds2 = bounds;
  Impl::RunCommandLaunchInfo launch_info(command, vsize);
  launch_info.beginExecute();
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    ARCCORE_KERNEL_CUDA_FUNC((Impl::doHierarchicalLaunchCudaHip<LoopBoundType, Lambda, RemainingArgs...>),
                             launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::HIP:
    ARCCORE_KERNEL_HIP_FUNC((Impl::doHierarchicalLaunchCudaHip<LoopBoundType, Lambda, RemainingArgs...>),
                            launch_info, func, bounds2, other_args...);
    break;
  case eExecutionPolicy::SYCL:
    ARCCORE_KERNEL_SYCL_FUNC((Impl::doHierarchicalLaunchSycl<LoopBoundType, Lambda, RemainingArgs...>{}),
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
 */
template <typename LoopBoundType, typename... RemainingArgs>
class ExtendedLaunchRunCommand
{
 public:

  ExtendedLaunchRunCommand(RunCommand& command, const LoopBoundType& bounds)
  : m_command(command)
  , m_bounds(bounds)
  {
  }
  ExtendedLaunchRunCommand(RunCommand& command, const LoopBoundType& bounds, const std::tuple<RemainingArgs...>& args)
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
 * \brief Classe pour gérer le lancement d'un noyau de calcul hiérarchique.
 */
template <typename LoopBoundType, typename... RemainingArgs>
class ExtendedLaunchLoop
{
 public:

  ExtendedLaunchLoop(const LoopBoundType& bounds, RemainingArgs... args)
  : m_bounds(bounds)
  , m_remaining_args(args...)
  {
  }
  LoopBoundType m_bounds;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename... RemainingArgs> auto
makeLaunch(const LoopBoundType& bounds, RemainingArgs... args)
-> ExtendedLaunchLoop<LoopBoundType, RemainingArgs...>
{
  return ExtendedLaunchLoop<LoopBoundType, RemainingArgs...>(bounds, args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
inline void operator<<(ExtendedLaunchRunCommand<LoopBoundType, RemainingArgs...>&& nr, const Lambda& f)
{
  if constexpr (sizeof...(RemainingArgs) > 0) {
    std::apply([&](auto... vs) { _doHierarchicalLaunch(nr.m_command, nr.m_bounds, f, vs...); }, nr.m_remaining_args);
  }
  else {
    _doHierarchicalLaunch(nr.m_command, nr.m_bounds, f);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Applique le fonctor \a func sur une boucle séqentielle.
 */
template <typename Lambda, typename... RemainingArgs> void
arccoreSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, const RemainingArgs&... remaining_args)
{
  Impl::WorkGroupSequentialForHelper::apply(0, bounds.nbGroup(), bounds, func, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Applique le fonctor \a func sur une boucle parallèle.
 */
template <typename Lambda, typename... RemainingArgs> void
arccoreParallelFor(WorkGroupLoopRange bounds, ForLoopRunInfo run_info,
                   const Lambda& func, const RemainingArgs&... remaining_args)
{
  auto sub_func = [=](Int32 begin_index, Int32 nb_loop) {
    Impl::WorkGroupSequentialForHelper::apply(begin_index, nb_loop, bounds, func, remaining_args...);
  };
  arccoreParallelFor(0, bounds.nbGroup(), run_info, sub_func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
