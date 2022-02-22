// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLoop.h                                            (C) 2000-2021 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une commande.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  impl::RunCommandLaunchInfo launch_info(command);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
#if defined(ARCANE_COMPILING_CUDA)
    {
      launch_info.beginExecute();
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
      // TODO: utiliser cudaLaunchKernel() à la place.
      impl::doDirectCUDALambdaArrayBounds<LoopBoundType<N>,Lambda> <<<b, t, 0, *s>>>(bounds,func);
    }
#else
    ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA compiler");
#endif
    break;
  case eExecutionPolicy::HIP:
#if defined(__HIP__)
    {
      launch_info.beginExecute();
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
      auto& loop_func = impl::doDirectCUDALambdaArrayBounds<LoopBoundType<N>,Lambda>;
      //impl::doDirectCUDALambdaArrayBounds<LoopBoundType<N>,Lambda> <<<b, t, 0, *s>>>(bounds,func);
      hipLaunchKernelGGL(loop_func, b, t, 0, *s, bounds, func);
    }
#else
    ARCANE_FATAL("Requesting HIP kernel execution but the kernel is not compiled with HIP compiler");
#endif
    break;
  case eExecutionPolicy::Sequential:
    launch_info.beginExecute();
    arcaneSequentialFor(bounds,func);
    break;
  case eExecutionPolicy::Thread:
    launch_info.beginExecute();
    arcaneParallelFor(bounds,func);
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
template<int N,typename Lambda> void
run(RunCommand& command,ArrayBounds<N> bounds,const Lambda& func)
{
  impl::_applyGenericLoop(command,SimpleLoopRanges(bounds),func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<int N,typename Lambda> void
run(RunCommand& command,SimpleLoopRanges<N> bounds,const Lambda& func)
{
  impl::_applyGenericLoop(command,bounds,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<int N,typename Lambda> void
run(RunCommand& command,ComplexLoopRanges<N> bounds,const Lambda& func)
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

template<int N> ArrayBoundRunCommand<N,SimpleLoopRanges<N>>
operator<<(RunCommand& command,const ArrayBounds<N>& bounds)
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,SimpleLoopRanges<N>>
operator<<(RunCommand& command,const SimpleLoopRanges<N>& bounds)
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,ComplexLoopRanges<N>>
operator<<(RunCommand& command,const ComplexLoopRanges<N>& bounds)
{
  return {command,bounds};
}

template<int N,template<int> class LoopBoundType,typename Lambda>
void operator<<(ArrayBoundRunCommand<N,LoopBoundType<N>>&& nr,const Lambda& f)
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
  A_FUNCINFO << ArrayBounds<N>(__VA_ARGS__) << [=] ARCCORE_HOST_DEVICE (ArrayBoundsIndex<N> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP1(iter_name, x1)                             \
  A_FUNCINFO << ArrayBounds<1>(x1) << [=] ARCCORE_HOST_DEVICE (ArrayBoundsIndex<1> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP2(iter_name, x1, x2)                             \
  A_FUNCINFO << ArrayBounds<2>(x1,x2) << [=] ARCCORE_HOST_DEVICE (ArrayBoundsIndex<2> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP3(iter_name, x1, x2, x3) \
  A_FUNCINFO << ArrayBounds<3>(x1,x2,x3) << [=] ARCCORE_HOST_DEVICE (ArrayBoundsIndex<3> iter_name )

//! Boucle sur accélérateur
#define RUNCOMMAND_LOOP4(iter_name, x1, x2, x3, x4)                        \
  A_FUNCINFO << ArrayBounds<4>(x1,x2,x3,x4) << [=] ARCCORE_HOST_DEVICE (ArrayBoundsIndex<4> iter_name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
