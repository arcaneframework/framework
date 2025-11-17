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
#ifndef ARCCORE_BASE_FORLOOPRANGES_H
#define ARCCORE_BASE_FORLOOPRANGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayBounds.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération pour une boucle.
 */
class ForLoopRange
{
 public:

  //! Créé un interval entre *[lower_bound,lower_bound+size[*
  ForLoopRange(Int32 lower_bound, Int32 size)
  : m_lower_bound(lower_bound)
  , m_size(size)
  {}
  //! Créé un interval entre *[0,size[*
  explicit(false) ForLoopRange(Int32 size)
  : m_lower_bound(0)
  , m_size(size)
  {}

 public:

  constexpr Int32 lowerBound() const { return m_lower_bound; }
  constexpr Int32 size() const { return m_size; }
  constexpr Int32 upperBound() const { return m_lower_bound + m_size; }

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
template <int N, typename IndexType_>
class SimpleForLoopRanges
{
  friend class ComplexForLoopRanges<N>;

 public:

  using ArrayBoundsType = ArrayBounds<typename MDDimType<N>::DimType>;
  using ArrayIndexType = ArrayBoundsType::MDIndexType;
  using LoopIndexType = ArrayIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Use 'LoopIndexType' instead") = LoopIndexType;

 public:

  explicit SimpleForLoopRanges(std::array<Int32, N> b)
  : m_bounds(b)
  {}
  explicit(false) SimpleForLoopRanges(ArrayBoundsType b)
  : m_bounds(b)
  {}

 public:

  template <Int32 I> static constexpr Int32 lowerBound() { return 0; }
  template <Int32 I> constexpr Int32 upperBound() const { return m_bounds.template constExtent<I>(); }
  template <Int32 I> constexpr Int32 extent() const { return m_bounds.template constExtent<I>(); }
  constexpr Int64 nbElement() const { return m_bounds.nbElement(); }
  constexpr ArrayIndexType getIndices(Int32 i) const { return m_bounds.getIndices(i); }

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
template <int N, typename IndexType_>
class ComplexForLoopRanges
{
 public:

  using ArrayBoundsType = ArrayBounds<typename MDDimType<N>::DimType>;
  using ArrayIndexType = ArrayBoundsType::MDIndexType;
  using LoopIndexType = ArrayIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Use 'LoopIndexType' instead") = LoopIndexType;

 public:

  ComplexForLoopRanges(ArrayBoundsType lower, ArrayBoundsType extents)
  : m_lower_bounds(lower.asStdArray())
  , m_extents(extents)
  {}
  explicit(false) ComplexForLoopRanges(const SimpleForLoopRanges<N>& bounds)
  : m_extents(bounds.m_bounds)
  {}

 public:

  template <Int32 I> constexpr Int32 lowerBound() const { return m_lower_bounds[I]; }
  template <Int32 I> constexpr Int32 upperBound() const { return m_lower_bounds[I] + m_extents.template constExtent<I>(); }
  template <Int32 I> constexpr Int32 extent() const { return m_extents.template constExtent<I>(); }
  constexpr Int64 nbElement() const { return m_extents.nbElement(); }
  constexpr ArrayIndexType getIndices(Int32 i) const
  {
    auto x = m_extents.getIndices(i);
    x.add(m_lower_bounds);
    return x;
  }
  constexpr ArrayBoundsType extents() const { return m_extents; }

 private:

  ArrayIndexType m_lower_bounds;
  ArrayBoundsType m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Créé un intervalle d'itération [0,n1[
inline SimpleForLoopRanges<1>
makeLoopRanges(Int32 n1)
{
  using BoundsType = SimpleForLoopRanges<1>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  return BoundsType(ArrayExtentType(n1));
}

//! Créé un intervalle d'itération [0,n1[, [0,n2[
inline SimpleForLoopRanges<2>
makeLoopRanges(Int32 n1, Int32 n2)
{
  using BoundsType = SimpleForLoopRanges<2>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  return BoundsType(ArrayExtentType(n1, n2));
}

//! Créé un intervalle d'itération [0,n1[, [0,n2[, [0,n3[
inline SimpleForLoopRanges<3>
makeLoopRanges(Int32 n1, Int32 n2, Int32 n3)
{
  using BoundsType = SimpleForLoopRanges<3>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  return BoundsType(ArrayExtentType(n1, n2, n3));
}

//! Créé un intervalle d'itération [0,n1[, [0,n2[, [0,n3[, [0,n4[
inline SimpleForLoopRanges<4>
makeLoopRanges(Int32 n1, Int32 n2, Int32 n3, Int32 n4)
{
  using BoundsType = SimpleForLoopRanges<4>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  return BoundsType(ArrayExtentType(n1, n2, n3, n4));
}

//! Créé un intervalle d'itération dans ℕ.
inline ComplexForLoopRanges<1>
makeLoopRanges(ForLoopRange n1)
{
  using BoundsType = ComplexForLoopRanges<1>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(n1.lowerBound()));
  BoundsType sizes(ArrayExtentType(n1.size()));
  return { lower_bounds, sizes };
}

//! Créé un intervalle d'itération dans ℕ².
inline ComplexForLoopRanges<2>
makeLoopRanges(ForLoopRange n1, ForLoopRange n2)
{
  using BoundsType = ComplexForLoopRanges<2>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(n1.lowerBound(), n2.lowerBound()));
  BoundsType sizes(ArrayExtentType(n1.size(), n2.size()));
  return { lower_bounds, sizes };
}

//! Créé un intervalle d'itération dans ℕ³.
inline ComplexForLoopRanges<3>
makeLoopRanges(ForLoopRange n1, ForLoopRange n2, ForLoopRange n3)
{
  using BoundsType = ComplexForLoopRanges<3>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(n1.lowerBound(), n2.lowerBound(), n3.lowerBound()));
  BoundsType sizes(ArrayExtentType(n1.size(), n2.size(), n3.size()));
  return { lower_bounds, sizes };
}

//! Créé un intervalle d'itération dans ℕ⁴.
inline ComplexForLoopRanges<4>
makeLoopRanges(ForLoopRange n1, ForLoopRange n2, ForLoopRange n3, ForLoopRange n4)
{
  using BoundsType = ComplexForLoopRanges<4>::ArrayBoundsType;
  using ArrayExtentType = BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(n1.lowerBound(), n2.lowerBound(), n3.lowerBound(), n4.lowerBound()));
  BoundsType sizes(ArrayExtentType(n1.size(), n2.size(), n3.size(), n4.size()));
  return { lower_bounds, sizes };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
