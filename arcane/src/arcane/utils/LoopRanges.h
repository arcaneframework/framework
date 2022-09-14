// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoopRanges.h                                                (C) 2000-2021 */
/*                                                                           */
/* Intervalles d'itérations pour les boucles.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LOOPRANGES_H
#define ARCANE_UTILS_LOOPRANGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayBounds.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération pour une boucle.
 */
class LoopRange
{
 public:
  //! Créé un interval entre *[lower_bound,lower_bound+size[*
  LoopRange(Int32 lower_bound,Int32 size)
  : m_lower_bound(lower_bound), m_size(size){}
  //! Créé un interval entre *[0,size[*
  LoopRange(Int32 size)
  : m_lower_bound(0), m_size(size){}
 public:
  constexpr Int32 lowerBound() const { return m_lower_bound; }
  constexpr Int32 size() const { return m_size; }
  constexpr Int32 upperBound() const { return m_lower_bound+m_size; }
 private:
  Int32 m_lower_bound;
  Int32 m_size;
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
  friend class ComplexLoopRanges<N>;
 public:
  using ArrayBoundsType = ArrayBounds<A_MDDIM(N)>;
  using ArrayBoundsIndexType = ArrayBoundsIndex<N>;
 public:
  typedef typename ArrayBoundsType::IndexType IndexType;
 public:
  SimpleLoopRanges(ArrayBoundsType b) : m_bounds(b){}
 public:
  constexpr Int32 lowerBound(int) const { return 0; }
  constexpr Int32 upperBound(int i) const { return m_bounds.extent(i); }
  constexpr Int32 extent(int i) const { return m_bounds.extent(i); }
  constexpr Int64 nbElement() const { return m_bounds.nbElement(); }
  constexpr ArrayBoundsIndexType getIndices(Int32 i) const { return m_bounds.getIndices(i); }
 private:
  ArrayBoundsType m_bounds;
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
  using ArrayBoundsType = ArrayBounds<A_MDDIM(N)>;
  using ArrayBoundsIndexType = ArrayBoundsIndex<N>;
 public:
  typedef typename ArrayBoundsType::IndexType IndexType;
 public:
  ComplexLoopRanges(ArrayBoundsType lower,ArrayBoundsType extents)
  : m_lower_bounds(lower.asStdArray()), m_extents(extents){}
  ComplexLoopRanges(const SimpleLoopRanges<N>& bounds)
  : m_extents(bounds.m_bounds){}
 public:
  constexpr Int32 lowerBound(int i) const { return m_lower_bounds[i]; }
  constexpr Int32 upperBound(int i) const { return m_lower_bounds[i]+m_extents.extent(i); }
  constexpr Int32 extent(int i) const { return m_extents.extent(i); }
  constexpr Int64 nbElement() const { return m_extents.nbElement(); }
  constexpr ArrayBoundsIndexType getIndices(Int32 i) const
  {
    auto x = m_extents.getIndices(i);
    x.add(m_lower_bounds);
    return x;
  }
 private:
  ArrayBoundsIndexType m_lower_bounds;
  ArrayBoundsType m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Créé un intervalle d'itération [0,n1[
inline SimpleLoopRanges<1>
makeLoopRanges(Int32 n1)
{
  SimpleLoopRanges<1>::ArrayBoundsType bounds(n1);
  return bounds;
}

//! Créé un intervalle d'itération [0,n1[,[0,n2[
inline SimpleLoopRanges<2>
makeLoopRanges(Int32 n1,Int32 n2)
{
  SimpleLoopRanges<2>::ArrayBoundsType bounds(n1,n2);
  return bounds;
}

//! Créé un intervalle d'itération [0,n1[,[0,n2[,[0,n3[
inline SimpleLoopRanges<3>
makeLoopRanges(Int32 n1,Int32 n2,Int32 n3)
{
  SimpleLoopRanges<3>::ArrayBoundsType bounds(n1,n2,n3);
  return bounds;
}

//! Créé un intervalle d'itération [0,n1[,[0,n2[,[0,n3[,[0,n4[
inline SimpleLoopRanges<4>
makeLoopRanges(Int32 n1,Int32 n2,Int32 n3,Int32 n4)
{
  SimpleLoopRanges<4>::ArrayBoundsType bounds(n1,n2,n3,n4);
  return bounds;
}

//! Créé un intervalle d'itération dans ℕ.
inline ComplexLoopRanges<1>
makeLoopRanges(LoopRange n1)
{
  ComplexLoopRanges<1>::ArrayBoundsType lower_bounds(n1.lowerBound());
  ComplexLoopRanges<1>::ArrayBoundsType sizes(n1.size());
  return {lower_bounds,sizes};
}

//! Créé un intervalle d'itération dans ℕ².
inline ComplexLoopRanges<2>
makeLoopRanges(LoopRange n1,LoopRange n2)
{
  ComplexLoopRanges<2>::ArrayBoundsType lower_bounds(n1.lowerBound(),n2.lowerBound());
  ComplexLoopRanges<2>::ArrayBoundsType sizes(n1.size(),n2.size());
  return {lower_bounds,sizes};
}

//! Créé un intervalle d'itération dans ℕ³.
inline ComplexLoopRanges<3>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3)
{
  ComplexLoopRanges<3>::ArrayBoundsType lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound());
  ComplexLoopRanges<3>::ArrayBoundsType sizes(n1.size(),n2.size(),n3.size());
  return {lower_bounds,sizes};
}

//! Créé un intervalle d'itération dans ℕ⁴.
inline ComplexLoopRanges<4>
makeLoopRanges(LoopRange n1,LoopRange n2,LoopRange n3,LoopRange n4)
{
  ComplexLoopRanges<4>::ArrayBoundsType lower_bounds(n1.lowerBound(),n2.lowerBound(),n3.lowerBound(),n4.lowerBound());
  ComplexLoopRanges<4>::ArrayBoundsType sizes(n1.size(),n2.size(),n3.size(),n4.size());
  return {lower_bounds,sizes};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle 1D.
template<template<int T> class LoopBoundType,typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<1> bounds,const Lambda& func)
{
  for( Int32 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    func(ArrayBoundsIndex<1>(i0));
}

//! Applique le fonctor \a func sur une boucle 2D.
template<template<int T> class LoopBoundType,typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<2> bounds,const Lambda& func)
{
  for( Int32 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int32 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      func(ArrayBoundsIndex<2>(i0,i1));
}

//! Applique le fonctor \a func sur une boucle 3D.
template<template<int T> class LoopBoundType,typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<3> bounds,const Lambda& func)
{
  for( Int32 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int32 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      for( Int32 i2 = bounds.lowerBound(2); i2 < bounds.upperBound(2); ++i2 )
        func(ArrayBoundsIndex<3>(i0,i1,i2));
}

//! Applique le fonctor \a func sur une boucle 4D.
template<template<int> class LoopBoundType,typename Lambda> inline void
arcaneSequentialFor(LoopBoundType<4> bounds,const Lambda& func)
{
  for( Int32 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int32 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      for( Int32 i2 = bounds.lowerBound(2); i2 < bounds.upperBound(2); ++i2 )
        for( Int32 i3 = bounds.lowerBound(3); i3 < bounds.upperBound(3); ++i3 )
          func(ArrayBoundsIndex<4>(i0,i1,i2,i3));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
