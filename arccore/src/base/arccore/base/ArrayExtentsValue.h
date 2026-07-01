// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExtentsValue.h                                         (C) 2000-2026 */
/*                                                                           */
/* Handling of N-dimensional array dimension values.                         */
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

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> constexpr ARCCORE_HOST_DEVICE T
fastmod(T a, T b)
{
  return a < b ? a : a - b * (a / b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information for a fixed dimension known at compile time.
 *
 * The dimension value is given as a template parameter
 */
template <Int32 Size, typename IndexType_ = Int32>
class ExtentValue
{
 public:

  using ExtentIndexType = IndexType_;

  static constexpr Int64 size() { return Size; };
  static constexpr ExtentIndexType v = Size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization for a dynamic dimension.
 *
 * The dimension value is stored in \a v.
 */
template <typename IndexType_>
class ExtentValue<DynExtent, IndexType_>
{
 public:

  using ExtentIndexType = IndexType_;

 public:

  constexpr Int64 size() const { return v; }

 public:

  ExtentIndexType v = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization to hold the dimensions of a 1-dimensional array.
 */
template <typename IndexType_, Int32 X0>
class ArrayExtentsValue<IndexType_, X0>
{
 public:

  using ExtentIndexType = IndexType_;
  using ExtentsType = ExtentsV<IndexType_, X0>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;
  using MDIndexType = MDIndex<1, IndexType_>;

  using LoopIndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = MDIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = MDIndexType;

  ArrayExtentsValue() = default;

  template <Int32 I> constexpr ExtentIndexType constExtent() const
  {
    static_assert(I == 0, "Invalid value for i (i==0)");
    return m_extent0.v;
  }

  constexpr std::array<ExtentIndexType, 1> asStdArray() const
  {
    return std::array<ExtentIndexType, 1>{ m_extent0.v };
  }

  template <typename OtherExtentIndexType>
  constexpr std::array<OtherExtentIndexType, 1> asOtherStdArray() const
  {
    return std::array<OtherExtentIndexType, 1>{
      static_cast<OtherExtentIndexType>(m_extent0.v)
    };
  }

  constexpr Int64 totalNbElement() const
  {
    return m_extent0.v;
  }

  constexpr MDIndexType getIndices(ExtentIndexType i) const
  {
    return { i };
  }

  constexpr ExtentIndexType extent0() const { return m_extent0.v; };

  //! List of dynamic dimensions
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<ExtentIndexType, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const ExtentIndexType> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
  }

  //! Constructs an instance with the N dynamic values.
  constexpr ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
  }

  constexpr std::array<ExtentIndexType, 0> _removeFirstExtent() const
  {
    return {};
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
  }

 protected:

  Impl::ExtentValue<X0, ExtentIndexType> m_extent0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization to hold the dimensions of a 2-dimensional array.
 */
template <typename IndexType_, Int32 X0, Int32 X1>
class ArrayExtentsValue<IndexType_, X0, X1>
{
 public:

  using ExtentIndexType = IndexType_;
  using ExtentsType = ExtentsV<IndexType_, X0, X1>;
  using MDIndexType = MDIndex<2, IndexType_>;
  using DynamicDimsType = ExtentsType::DynamicDimsType;

  using LoopIndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = MDIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ExtentIndexType constExtent() const
  {
    static_assert(I >= 0 && I < 2, "Invalid value for I (0<=I<2)");
    if (I == 0)
      return m_extent0.v;
    return m_extent1.v;
  }

  constexpr std::array<ExtentIndexType, 2> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v };
  }

  template <typename OtherExtentIndexType>
  constexpr std::array<OtherExtentIndexType, 2> asOtherStdArray() const
  {
    return std::array<OtherExtentIndexType, 2>{ m_extent0.v, m_extent1.v };
  }

  constexpr Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size();
  }

  constexpr MDIndexType getIndices(ExtentIndexType i) const
  {
    ExtentIndexType i0 = i / m_extent1.v;
    ExtentIndexType i1 = i % m_extent1.v;
    return { i0, i1 };
  }

  constexpr ExtentIndexType extent0() const { return m_extent0.v; };
  constexpr ExtentIndexType extent1() const { return m_extent1.v; };

  //! List of dynamic dimensions
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<ExtentIndexType, ExtentsType::nb_dynamic> x = {};
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      x[i++] = m_extent0.v;
    if constexpr (X1 == DynExtent)
      x[i++] = m_extent1.v;
    return DynamicDimsType(x);
  }

 protected:

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const ExtentIndexType> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
    if constexpr (X1 == DynExtent)
      m_extent1.v = extents[1];
  }

  //! Constructs an instance with the N dynamic values.
  constexpr ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
    if constexpr (X1 == DynExtent)
      m_extent1.v = dims[i++];
  }

  constexpr std::array<ExtentIndexType, 1> _removeFirstExtent() const
  {
    return std::array<ExtentIndexType, 1>{ m_extent1.v };
  }

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] MDIndexType idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(), m_extent0.v);
    ARCCORE_CHECK_AT(idx.id1(), m_extent1.v);
  }

 protected:

  Impl::ExtentValue<X0, ExtentIndexType> m_extent0;
  Impl::ExtentValue<X1, ExtentIndexType> m_extent1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization to hold the dimensions of a 3-dimensional array.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2>
class ArrayExtentsValue<IndexType_, X0, X1, X2>
{
 public:

  using ExtentIndexType = IndexType_;
  using ExtentsType = ExtentsV<IndexType_, X0, X1, X2>;
  using MDIndexType = MDIndex<3, IndexType_>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;

  using LoopIndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = MDIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ExtentIndexType constExtent() const
  {
    static_assert(I >= 0 && I < 3, "Invalid value for I (0<=I<3)");
    if (I == 0)
      return m_extent0.v;
    if (I == 1)
      return m_extent1.v;
    return m_extent2.v;
  }

  constexpr std::array<ExtentIndexType, 3> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v, m_extent2.v };
  }

  template <typename OtherExtentIndexType>
  constexpr std::array<OtherExtentIndexType, 3> asOtherStdArray() const
  {
    return std::array<OtherExtentIndexType, 3>{ m_extent0.v, m_extent1.v, m_extent2.v };
  }

  constexpr Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size() * m_extent2.size();
  }

  constexpr MDIndexType getIndices(ExtentIndexType i) const
  {
    ExtentIndexType i0 = i / (m_extent1.v * m_extent2.v);
    i %= (m_extent1.v * m_extent2.v);
    ExtentIndexType i1 = i / m_extent2.v;
    ExtentIndexType i2 = i % m_extent2.v;
    return { i0, i1, i2 };
  }

  constexpr ExtentIndexType extent0() const { return m_extent0.v; };
  constexpr ExtentIndexType extent1() const { return m_extent1.v; };
  constexpr ExtentIndexType extent2() const { return m_extent2.v; };

  //! List of dynamic dimensions
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<ExtentIndexType, ExtentsType::nb_dynamic> x = {};
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

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const ExtentIndexType> extents)
  {
    if constexpr (X0 == DynExtent)
      m_extent0.v = extents[0];
    if constexpr (X1 == DynExtent)
      m_extent1.v = extents[1];
    if constexpr (X2 == DynExtent)
      m_extent2.v = extents[2];
  }

  //! Constructs an instance with N dynamic values.
  constexpr ArrayExtentsValue([[maybe_unused]] DynamicDimsType dims)
  {
    Int32 i = 0;
    if constexpr (X0 == DynExtent)
      m_extent0.v = dims[i++];
    if constexpr (X1 == DynExtent)
      m_extent1.v = dims[i++];
    if constexpr (X2 == DynExtent)
      m_extent2.v = dims[i++];
  }

  constexpr std::array<ExtentIndexType, 2> _removeFirstExtent() const
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

  Impl::ExtentValue<X0, ExtentIndexType> m_extent0;
  Impl::ExtentValue<X1, ExtentIndexType> m_extent1;
  Impl::ExtentValue<X2, ExtentIndexType> m_extent2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Specialization to hold the dimensions of a 4-dimensional array.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2, Int32 X3>
class ArrayExtentsValue<IndexType_, X0, X1, X2, X3>
{
 public:

  using ExtentIndexType = IndexType_;
  using ExtentsType = ExtentsV<IndexType_, X0, X1, X2, X3>;
  using MDIndexType = MDIndex<4, IndexType_>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;

  using LoopIndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = MDIndexType;
  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayExtentsValue() = default;

 public:

  template <Int32 I> constexpr ARCCORE_HOST_DEVICE ExtentIndexType constExtent() const
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

  constexpr ARCCORE_HOST_DEVICE std::array<ExtentIndexType, 4> asStdArray() const
  {
    return { m_extent0.v, m_extent1.v, m_extent2.v, m_extent3.v };
  }

  template <typename OtherExtentIndexType>
  constexpr std::array<OtherExtentIndexType, 4> asOtherStdArray() const
  {
    return std::array<OtherExtentIndexType, 4>{ m_extent0.v, m_extent1.v, m_extent2.v, m_extent3.v };
  }

  constexpr Int64 totalNbElement() const
  {
    return m_extent0.size() * m_extent1.size() * m_extent2.size() * m_extent3.size();
  }

  constexpr MDIndexType getIndices(ExtentIndexType i) const
  {
    // Compute base indices
    ExtentIndexType i3 = Impl::fastmod(i, m_extent3.v);
    ExtentIndexType fac = m_extent3.v;
    ExtentIndexType i2 = Impl::fastmod(i / fac, m_extent2.v);
    fac *= m_extent2.v;
    ExtentIndexType i1 = Impl::fastmod(i / fac, m_extent1.v);
    fac *= m_extent1.v;
    ExtentIndexType i0 = i / fac;
    return { i0, i1, i2, i3 };
  }

  constexpr ExtentIndexType extent0() const { return m_extent0.v; };
  constexpr ExtentIndexType extent1() const { return m_extent1.v; };
  constexpr ExtentIndexType extent2() const { return m_extent2.v; };
  constexpr ExtentIndexType extent3() const { return m_extent3.v; };

  //! List of dynamic dimensions
  constexpr DynamicDimsType dynamicExtents() const
  {
    std::array<ExtentIndexType, ExtentsType::nb_dynamic> x = {};
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

  explicit ARCCORE_HOST_DEVICE ArrayExtentsValue(SmallSpan<const ExtentIndexType> extents)
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

  //! Constructs an instance with N dynamic values.
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

  constexpr std::array<ExtentIndexType, 3> _removeFirstExtent() const
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

  Impl::ExtentValue<X0, ExtentIndexType> m_extent0;
  Impl::ExtentValue<X1, ExtentIndexType> m_extent1;
  Impl::ExtentValue<X2, ExtentIndexType> m_extent2;
  Impl::ExtentValue<X3, ExtentIndexType> m_extent3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
