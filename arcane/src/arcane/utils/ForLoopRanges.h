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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle 1D.
template <typename IndexType, template <int T, typename> class LoopBoundType,
          typename Lambda, typename... ReducerArgs>
inline void
arcaneSequentialFor(LoopBoundType<1, IndexType> bounds, const Lambda& func, ReducerArgs... reducer_args)
{
  for (Int32 i0 = bounds.template lowerBound<0>(); i0 < bounds.template upperBound<0>(); ++i0)
    func(MDIndex<1>(i0), reducer_args...);
  ::Arcane::impl::HostReducerHelper::applyReducerArgs(reducer_args...);
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
