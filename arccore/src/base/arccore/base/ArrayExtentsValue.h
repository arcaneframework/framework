// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExtentsValue.h                                         (C) 2000-2025 */
/*                                                                           */
/* Gestion de valeurs des dimensions des tableaux N-dimensions.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYEXTENTSVALUE_H
#define ARCCORE_BASE_ARRAYEXTENTSVALUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/MDIndex.h"
#include "arccore/base/ArrayLayout.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> constexpr ARCCORE_HOST_DEVICE
T
fastmod(T a, T b)
{
  return a < b ? a : a - b * (a / b);
}

/*!
 * \brief Informations pour une dimension fixe connue à la compilation.
 *
 * La valeur de la dimension est donnée en paramètre template
 */
template <Int32 Size, typename IndexType_ = Int32>
class ExtentValue
{
 public:

  static constexpr Int64 size() { return Size; };
  static constexpr Int32 v = Size;
};

/*!
 * \brief Spécialisation pour une dimension dynamique.
 *
 * La valeur de la dimension est conservée dans \a v.
 */
template <typename IndexType_>
class ExtentValue<DynExtent, IndexType_>
{
 public:

  constexpr Int64 size() const { return v; }

 public:

  IndexType_ v = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour contenir les dimensions d'un tableau à 1 dimension.
 */
template <typename IndexType_, Int32 X0>
class ArrayExtentsValue<IndexType_, X0>
{
 public:

  using ExtentsType = ExtentsV<IndexType_, X0>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;
  using MDIndexType = MDIndex<1>;
  using LoopIndexType = MDIndex<1>;

  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'LoopIndexType' or 'MDIndexType' instead") = LoopIndexType;

  ArrayExtentsValue() = default;

  template <Int32 I> constexpr ARCCORE_HOST_DEVICE Int32 constExtent() const
  {
    static_assert(I == 0, "Invalid value for i (i==0)");
    return m_extent0.v;
  }

  constexpr ARCCORE_HOST_DEVICE std::array<Int32, 1> asStdArray() const
  {
    return std::array<Int32, 1>{ m_extent0.v };
  }

  constexpr ARCCORE_HOST_DEVICE Int64 totalNbElement() const
  {
    return m_extent0.v;
  }

  constexpr ARCCORE_HOST_DEVICE MDIndexType getIndices(Int32 i) const
  {
    return { i };
  }

  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extent0.v; };

  //! Liste des dimensions dynamiques
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<Int32, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const Int32> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
  }

  //! Construit une instance avec les N valeurs dynamiques.
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
  }

  constexpr std::array<Int32, 0> _removeFirstExtent() const
  {
    return {};
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
  }

 protected:

  impl::ExtentValue<X0> m_extent0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour contenir les dimensions d'un tableau à 2 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1>
class ArrayExtentsValue<IndexType_, X0, X1>
{
 public:

  using ExtentsType = ExtentsV<IndexType_, X0, X1>;
  using MDIndexType = MDIndex<2>;
  using LoopIndexType = MDIndex<2>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;

  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'LoopIndexType' or 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ARCCORE_HOST_DEVICE Int32 constExtent() const
  {
    static_assert(I >= 0 && I < 2, "Invalid value for I (0<=I<2)");
    if (I == 0)
      return m_extent0.v;
    return m_extent1.v;
  }

  constexpr ARCCORE_HOST_DEVICE std::array<Int32, 2> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v };
  }

  constexpr ARCCORE_HOST_DEVICE Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size();
  }

  constexpr ARCCORE_HOST_DEVICE MDIndexType getIndices(Int32 i) const
  {
    Int32 i1 = impl::fastmod(i, m_extent1.v);
    Int32 i0 = i / m_extent1.v;
    return { i0, i1 };
  }

  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extent0.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extent1.v; };

  //! Liste des dimensions dynamiques
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<Int32, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    if constexpr (X1 == DynExtent)
      x[i++] = m_extent1.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const Int32> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
    if constexpr (X1 == DynExtent)
      m_extent1.v = extents[1];
  }

  //! Construit une instance avec les N valeurs dynamiques.
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
    if constexpr (X1 == DynExtent)
      m_extent1.v = dims[i++];
  }

  constexpr std::array<Int32, 1> _removeFirstExtent() const
  {
    return std::array<Int32, 1>{ m_extent1.v };
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
    ARCCORE_CHECK_AT(idx.id1(), m_extent1.v);
  }

 protected:

  impl::ExtentValue<X0> m_extent0;
  impl::ExtentValue<X1> m_extent1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour contenir les dimensions d'un tableau à 3 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2>
class ArrayExtentsValue<IndexType_, X0, X1, X2>
{
 public:

  using ExtentsType = ExtentsV<IndexType_, X0, X1, X2>;
  using MDIndexType = MDIndex<3>;
  using LoopIndexType = MDIndex<3>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;

  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'LoopIndexType' or 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ARCCORE_HOST_DEVICE Int32 constExtent() const
  {
    static_assert(I >= 0 && I < 3, "Invalid value for I (0<=I<3)");
    if (I == 0)
      return m_extent0.v;
    if (I == 1)
      return m_extent1.v;
    return m_extent2.v;
  }

  constexpr ARCCORE_HOST_DEVICE std::array<Int32, 3> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v, m_extent2.v };
  }

  constexpr ARCCORE_HOST_DEVICE Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size() * m_extent2.size();
  }

  constexpr ARCCORE_HOST_DEVICE MDIndexType getIndices(Int32 i) const
  {
    Int32 i2 = impl::fastmod(i, m_extent2.v);
    Int32 fac = m_extent2.v;
    Int32 i1 = impl::fastmod(i / fac, m_extent1.v);
    fac *= m_extent1.v;
    Int32 i0 = i / fac;
    return { i0, i1, i2 };
  }

  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extent0.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extent1.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent2() const { return m_extent2.v; };

  //! Liste des dimensions dynamiques
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<Int32, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    if constexpr (X1 == DynExtent)
      x[i++] = m_extent1.v;
    if constexpr (X2 == DynExtent)
      x[i++] = m_extent2.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const Int32> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
    if constexpr (X1 == DynExtent)
      m_extent1.v = extents[1];
    if constexpr (X2 == DynExtent)
      m_extent2.v = extents[2];
  }

  //! Construit une instance avec les N valeurs dynamiques.
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
    if constexpr (X1 == DynExtent)
      m_extent1.v = dims[i++];
    if constexpr (X2 == DynExtent)
      m_extent2.v = dims[i++];
  }

  constexpr std::array<Int32, 2> _removeFirstExtent() const
  {
    return { m_extent1.v, m_extent2.v };
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
    ARCCORE_CHECK_AT(idx.id1(), m_extent1.v);
    ARCCORE_CHECK_AT(idx.id2(), m_extent2.v);
  }

 protected:

  impl::ExtentValue<X0> m_extent0;
  impl::ExtentValue<X1> m_extent1;
  impl::ExtentValue<X2> m_extent2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour contenir les dimensions d'un tableau à 4 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2, Int32 X3>
class ArrayExtentsValue<IndexType_, X0, X1, X2, X3>
{
 public:

  using ExtentsType = ExtentsV<IndexType_, X0, X1, X2, X3>;
  using MDIndexType = MDIndex<4>;
  using LoopIndexType = MDIndex<4>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;

  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'LoopIndexType' or 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ARCCORE_HOST_DEVICE Int32 constExtent() const
  {
    static_assert(I >= 0 && I < 4, "Invalid value for I (0<=I<4)");
    if (I == 0)
      return m_extent0.v;
    if (I == 1)
      return m_extent1.v;
    if (I == 2)
      return m_extent2.v;
    return m_extent3.v;
  }

  constexpr ARCCORE_HOST_DEVICE std::array<Int32, 4> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v, m_extent2.v, m_extent3.v };
  }

  constexpr ARCCORE_HOST_DEVICE Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size() * m_extent2.size() * m_extent3.size();
  }

  constexpr ARCCORE_HOST_DEVICE MDIndexType getIndices(Int32 i) const
  {
    // Compute base indices
    Int32 i3 = impl::fastmod(i, m_extent3.v);
    Int32 fac = m_extent3.v;
    Int32 i2 = impl::fastmod(i / fac, m_extent2.v);
    fac *= m_extent2.v;
    Int32 i1 = impl::fastmod(i / fac, m_extent1.v);
    fac *= m_extent1.v;
    Int32 i0 = i / fac;
    return { i0, i1, i2, i3 };
  }

  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extent0.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extent1.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent2() const { return m_extent2.v; };
  constexpr ARCCORE_HOST_DEVICE Int32 extent3() const { return m_extent3.v; };

  //! Liste des dimensions dynamiques
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<Int32, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    if constexpr (X1 == DynExtent)
      x[i++] = m_extent1.v;
    if constexpr (X2 == DynExtent)
      x[i++] = m_extent2.v;
    if constexpr (X3 == DynExtent)
      x[i++] = m_extent3.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const Int32> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
    if constexpr (X1 == DynExtent)
      m_extent1.v = extents[1];
    if constexpr (X2 == DynExtent)
      m_extent2.v = extents[2];
    if constexpr (X3 == DynExtent)
      m_extent3.v = extents[3];
  }

  //! Construit une instance avec les N valeurs dynamiques.
  ARCCORE_HOST_DEVICE ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
    if constexpr (X1 == DynExtent)
      m_extent1.v = dims[i++];
    if constexpr (X2 == DynExtent)
      m_extent2.v = dims[i++];
    if constexpr (X3 == DynExtent)
      m_extent3.v = dims[i++];
  }

  constexpr std::array<Int32, 3> _removeFirstExtent() const
  {
    return { m_extent1.v, m_extent2.v, m_extent3.v };
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
    ARCCORE_CHECK_AT(idx.id1(), m_extent1.v);
    ARCCORE_CHECK_AT(idx.id2(), m_extent2.v);
    ARCCORE_CHECK_AT(idx.id3(), m_extent3.v);
  }

 protected:

  impl::ExtentValue<X0> m_extent0;
  impl::ExtentValue<X1> m_extent1;
  impl::ExtentValue<X2> m_extent2;
  impl::ExtentValue<X3> m_extent3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
