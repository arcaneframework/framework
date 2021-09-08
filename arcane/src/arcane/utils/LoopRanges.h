// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoopRanges.h                                                (C) 2000-2021 */
/*                                                                           */
/* Intervalles d'itérations pour les boucles.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LOOPRANGE_H
#define ARCANE_UTILS_LOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayBounds.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interval d'itération simple.
 *
 * Les indices de début pour chaque dimension commencent à 0.
 */
template <int N>
class SimpleLoopRanges
{
 public:
  typedef typename ArrayBounds<N>::IndexType IndexType;
 public:
  SimpleLoopRanges(ArrayBounds<N> b) : m_bounds(b){}
 public:
  constexpr Int64 lowerBound(int) const { return 0; }
  constexpr Int64 upperBound(int i) const { return m_bounds.extent(i); }
  constexpr Int64 extent(int i) const { return m_bounds.extent(i); }
  constexpr Int64 nbElement() const { return m_bounds.nbElement(); }
  constexpr ArrayBoundsIndex<N> getIndices(Int64 i) const { return m_bounds.getIndices(i); }
 private:
  ArrayBounds<N> m_bounds;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interval d'itération complexe.
 *
 * Les indices de début pour chaque dimension sont spécifiés \a lower et
 * le nombre d'éléments dans chaque dimension par \a extents.
 */
template <int N>
class ComplexLoopRanges
{
 public:
  typedef typename ArrayBounds<N>::IndexType IndexType;
 public:
  ComplexLoopRanges(ArrayBounds<N> lower,ArrayBounds<N> extents)
  : m_lower_bounds(lower.asStdArray()), m_extents(extents){}
 public:
  constexpr Int64 lowerBound(int i) const { return m_lower_bounds[i]; }
  constexpr Int64 upperBound(int i) const { return m_lower_bounds[i]+m_extents.extent(i); }
  constexpr Int64 extent(int i) const { return m_extents.extent(i); }
  constexpr Int64 nbElement() const { return m_extents.nbElement(); }
  constexpr ArrayBoundsIndex<N> getIndices(Int64 i) const
  {
    auto x = m_extents.getIndices(i);
    x.add(m_lower_bounds);
    return x;
  }
 private:
  ArrayBoundsIndex<N> m_lower_bounds;
  ArrayBounds<N> m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline SimpleLoopRanges<1>
makeLoopRanges(Int64 n1)
{
  ArrayBounds<1> bounds(n1);
  return bounds;
}

inline SimpleLoopRanges<2>
makeLoopRanges(Int64 n1,Int64 n2)
{
  ArrayBounds<2> bounds(n1,n2);
  return bounds;
}

inline SimpleLoopRanges<3>
makeLoopRanges(Int64 n1,Int64 n2,Int64 n3)
{
  ArrayBounds<3> bounds(n1,n2,n3);
  return bounds;
}

inline SimpleLoopRanges<4>
makeLoopRanges(Int64 n1,Int64 n2,Int64 n3,Int64 n4)
{
  ArrayBounds<4> bounds(n1,n2,n3,n4);
  return bounds;
}

inline ComplexLoopRanges<1>
makeLoopRanges(LoopRange n1)
{
  ArrayBounds<1> lower_bounds(n1.lowerBound());
  ArrayBounds<1> sizes(n1.size());
  return {lower_bounds,sizes};
}

inline ComplexLoopRanges<2>
makeLoopRanges(LoopRange n1,LoopRange n2)
{
  ArrayBounds<2> lower_bounds(n1.lowerBound(),n2.lowerBound());
  ArrayBounds<2> sizes(n1.size(),n2.size());
  return {lower_bounds,sizes};
}

inline ComplexLoopRanges<3>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3)
{
  ArrayBounds<3> lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound());
  ArrayBounds<3> sizes(n1.size(),n2.size(),n3.size());
  return {lower_bounds,sizes};
}

inline ComplexLoopRanges<4>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3,LoopRange n4)
{
  ArrayBounds<4> lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound(),n4.lowerBound());
  ArrayBounds<4> sizes(n1.size(),n2.size(),n3.size(),n4.size());
  return {lower_bounds,sizes};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
