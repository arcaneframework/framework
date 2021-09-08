// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
  A_FUNCINFO << bounds << [=] ARCCORE_HOST_DEVICE (decltype(bounds) :: IndexType iter_name )

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
