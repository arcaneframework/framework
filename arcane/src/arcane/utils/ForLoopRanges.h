// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ForLoopRanges.h                                             (C) 2000-2025 */
/*                                                                           */
/* Intervalles d'itérations pour les boucles.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FORLOOPRANGES_H
#define ARCANE_UTILS_FORLOOPRANGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ForLoopRanges.h"
#include "arccore/common/SequentialFor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle 1D.
template <typename IndexType, template <int T, typename> class LoopBoundType,
          typename Lambda, typename... RemainingArgs>
void arcaneSequentialFor(LoopBoundType<1, IndexType> bounds, const Lambda& func, RemainingArgs... remaining_args)
{
  Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtBegin(remaining_args...);
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    func(MDIndex<1>(i0), remaining_args...);
  Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtEnd(remaining_args...);
}

//! Applique le fonctor \a func sur une boucle 2D.
template <typename IndexType, template <int T, typename> class LoopBoundType, typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<2, IndexType> bounds, const Lambda& func)
{
  arccoreSequentialFor(bounds,func);
}

//! Applique le fonctor \a func sur une boucle 3D.
template <typename IndexType, template <int T, typename> class LoopBoundType, typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<3, IndexType> bounds, const Lambda& func)
{
  arccoreSequentialFor(bounds,func);
}

//! Applique le fonctor \a func sur une boucle 4D.
template <typename IndexType, template <int, typename> class LoopBoundType, typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<4, IndexType> bounds, const Lambda& func)
{
  arccoreSequentialFor(bounds,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
