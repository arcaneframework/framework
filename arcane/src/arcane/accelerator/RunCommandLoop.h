// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLoop.h                                            (C) 2000-2023 */
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

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique la lambda \a func sur une boucle \a bounds
 */
template<int N,template<int T> class LoopBoundType,typename Lambda> void
_applyGenericLoop(RunCommand& command,LoopBoundType<N> bounds,const Lambda& func)
{
  Int64 vsize = bounds.nbElement();
  if (vsize==0)
    return;
  impl::RunCommandLaunchInfo launch_info(command,vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.beginExecute();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info,ARCANE_KERNEL_CUDA_FUNC(impl::doDirectGPULambdaArrayBounds)<LoopBoundType<N>,Lambda>,func,bounds);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info,ARCANE_KERNEL_HIP_FUNC(impl::doDirectGPULambdaArrayBounds)<LoopBoundType<N>,Lambda>,func,bounds);
    break;
  case eExecutionPolicy::Sequential:
    arcaneSequentialFor(bounds,func);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelFor(bounds,launch_info.computeParallelLoopOptions(vsize),func);
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'",exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<typename ExtentType,typename Lambda> void
run(RunCommand& command,ArrayBounds<ExtentType> bounds,const Lambda& func)
{
  impl::_applyGenericLoop(command,SimpleForLoopRanges<ExtentType::rank()>(bounds),func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<int N,typename Lambda> void
run(RunCommand& command,SimpleForLoopRanges<N> bounds,const Lambda& func)
{
  impl::_applyGenericLoop(command,bounds,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<int N,typename Lambda> void
run(RunCommand& command,ComplexForLoopRanges<N> bounds,const Lambda& func)
{
  impl::_applyGenericLoop(command,bounds,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int N,typename LoopBoundType>
class ArrayBoundRunCommand
{
 public:
  ArrayBoundRunCommand(RunCommand& command,const LoopBoundType& bounds)
  : m_command(command), m_bounds(bounds)
  {
  }
  RunCommand& m_command;
  LoopBoundType m_bounds;
};

template<typename ExtentType> auto
operator<<(RunCommand& command,const ArrayBounds<ExtentType>& bounds)
  -> ArrayBoundRunCommand<ExtentType::rank(),SimpleForLoopRanges<ExtentType::rank()>>
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,SimpleForLoopRanges<N>>
operator<<(RunCommand& command,const SimpleForLoopRanges<N>& bounds)
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,ComplexForLoopRanges<N>>
operator<<(RunCommand& command,const ComplexForLoopRanges<N>& bounds)
{
  return {command,bounds};
}

template<int N,template<int> class ForLoopBoundType,typename Lambda>
void operator<<(ArrayBoundRunCommand<N,ForLoopBoundType<N>>&& nr,const Lambda& f)
{
  run(nr.m_command,nr.m_bounds,f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP(iter_name, bounds)                              \
  A_FUNCINFO << bounds << [=] ARCCORE_HOST_DEVICE (typename decltype(bounds) :: IndexType iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOPN(iter_name, N, ...)                           \
  A_FUNCINFO << Arcane::ArrayBounds<typename Arcane::MDDimType<N>::DimType>(__VA_ARGS__) << [=] ARCCORE_HOST_DEVICE (Arcane::ArrayIndex<N> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP1(iter_name, x1)                             \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim1>(x1) << [=] ARCCORE_HOST_DEVICE (Arcane::ArrayIndex<1> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP2(iter_name, x1, x2)                             \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim2>(x1,x2) << [=] ARCCORE_HOST_DEVICE (Arcane::ArrayIndex<2> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP3(iter_name, x1, x2, x3) \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim3>(x1,x2,x3) << [=] ARCCORE_HOST_DEVICE (Arcane::ArrayIndex<3> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP4(iter_name, x1, x2, x3, x4)                        \
  A_FUNCINFO << Arcane::ArrayBounds<MDDim4>(x1,x2,x3,x4) << [=] ARCCORE_HOST_DEVICE (Arcane::ArrayIndex<4> iter_name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
