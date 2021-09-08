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

template<int N> ArrayBoundRunCommand<N,impl::SimpleLoopRanges<N>>
operator<<(RunCommand& command,const ArrayBounds<N>& bounds)
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,impl::SimpleLoopRanges<N>>
operator<<(RunCommand& command,const impl::SimpleLoopRanges<N>& bounds)
{
  return {command,bounds};
}

template<int N> ArrayBoundRunCommand<N,impl::ComplexLoopRanges<N>>
operator<<(RunCommand& command,const impl::ComplexLoopRanges<N>& bounds)
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
/*!
 * \brief Intervalle d'itération pour une boucle.
 */
struct LoopRange
{
 public:
  //! Créé un interval entre *[lower_bound,lower_bound+size[*
  LoopRange(Int64 lower_bound,Int64 size)
  : m_lower_bound(lower_bound), m_size(size){}
  //! Créé un interval entre *[0,size[*
  LoopRange(Int64 size)
  : m_lower_bound(0), m_size(size){}
 public:
  constexpr Int64 lowerBound() const { return m_lower_bound; }
  constexpr Int64 size() const { return m_size; }
  constexpr Int64 upperBound() const { return m_lower_bound+m_size; }
 private:
  Int64 m_lower_bound;
  Int64 m_size;
};

inline impl::SimpleLoopRanges<1>
makeLoopRanges(Int64 n1)
{
  ArrayBounds<1> bounds(n1);
  return bounds;
}

inline impl::SimpleLoopRanges<2>
makeLoopRanges(Int64 n1,Int64 n2)
{
  ArrayBounds<2> bounds(n1,n2);
  return bounds;
}

inline impl::SimpleLoopRanges<3>
makeLoopRanges(Int64 n1,Int64 n2,Int64 n3)
{
  ArrayBounds<3> bounds(n1,n2,n3);
  return bounds;
}

inline impl::SimpleLoopRanges<4>
makeLoopRanges(Int64 n1,Int64 n2,Int64 n3,Int64 n4)
{
  ArrayBounds<4> bounds(n1,n2,n3,n4);
  return bounds;
}

inline impl::ComplexLoopRanges<1>
makeLoopRanges(LoopRange n1)
{
  ArrayBounds<1> lower_bounds(n1.lowerBound());
  ArrayBounds<1> sizes(n1.size());
  return {lower_bounds,sizes};
}

inline impl::ComplexLoopRanges<2>
makeLoopRanges(LoopRange n1,LoopRange n2)
{
  ArrayBounds<2> lower_bounds(n1.lowerBound(),n2.lowerBound());
  ArrayBounds<2> sizes(n1.size(),n2.size());
  return {lower_bounds,sizes};
}

inline impl::ComplexLoopRanges<3>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3)
{
  ArrayBounds<3> lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound());
  ArrayBounds<3> sizes(n1.size(),n2.size(),n3.size());
  return {lower_bounds,sizes};
}

inline impl::ComplexLoopRanges<4>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3,LoopRange n4)
{
  ArrayBounds<4> lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound(),n4.lowerBound());
  ArrayBounds<4> sizes(n1.size(),n2.size(),n3.size(),n4.size());
  return {lower_bounds,sizes};
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
