// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialFor.h                                             (C) 2000-2025 */
/*                                                                           */
/* Gestion des boucles for en séquentiel.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_SEQUENTIALFOR_H
#define ARCCORE_COMMON_SEQUENTIALFOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/HostKernelRemainingArgsHelper.h"
#include "arccore/base/ForLoopRanges.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le functor \a func sur une boucle 1D.
template <typename IndexType, template <int T, typename> class LoopBoundType,
          typename Lambda, typename... RemainingArgs>
void arccoreSequentialFor(LoopBoundType<1, IndexType> bounds, const Lambda& func,
                          RemainingArgs... remaining_args)
{
  Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    func(MDIndex<1>(i0), remaining_args...);
  Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le functor \a func sur une boucle 2D.
template <typename IndexType, template <int T, typename> class LoopBoundType, typename Lambda> void
arccoreSequentialFor(LoopBoundType<2, IndexType> bounds, const Lambda& func)
{
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    for (Int32 i1 = bounds.template lowerBound<1>(); i1 < bounds.template upperBound<1>(); ++i1)
      func(MDIndex<2>(i0, i1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le functor \a func sur une boucle 3D.
template <typename IndexType, template <int T, typename> class LoopBoundType, typename Lambda> void
arccoreSequentialFor(LoopBoundType<3, IndexType> bounds, const Lambda& func)
{
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    for (Int32 i1 = bounds.template lowerBound<1>(); i1 < bounds.template upperBound<1>(); ++i1)
      for (Int32 i2 = bounds.template lowerBound<2>(); i2 < bounds.template upperBound<2>(); ++i2)
        func(MDIndex<3>(i0, i1, i2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le functor \a func sur une boucle 4D.
template <typename IndexType, template <int, typename> class LoopBoundType, typename Lambda> void
arccoreSequentialFor(LoopBoundType<4, IndexType> bounds, const Lambda& func)
{
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    for (Int32 i1 = bounds.template lowerBound<1>(); i1 < bounds.template upperBound<1>(); ++i1)
      for (Int32 i2 = bounds.template lowerBound<2>(); i2 < bounds.template upperBound<2>(); ++i2)
        for (Int32 i3 = bounds.template lowerBound<3>(); i3 < bounds.template upperBound<3>(); ++i3)
          func(MDIndex<4>(i0, i1, i2, i3));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
