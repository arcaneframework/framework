// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDSpan.h                                                    (C) 2000-2026 */
/*                                                                           */
/* View on a multi-dimensional array for numeric types.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MDSPAN_H
#define ARCCORE_BASE_MDSPAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayExtents.h"
#include "arccore/base/ArrayBounds.h"
#include "arccore/base/NumericTraits.h"
#include "arccore/base/ArrayLayout.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Base class for multidimensional views.
 *
 * This class is inspired by the std::mdspan class currently being defined
 * (see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0009r12.html)
 *
 * This class is used to manage views on arrays such as
 * NumArray. The methods of this class are accessible on the accelerator.
 *
 * For more information, refer to the page \ref arcanedoc_core_types_numarray.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpan
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArray<UnqualifiedValueType, Extents, LayoutPolicy>;
  // For MDSpan<const T> to access MDSpan<T>
  friend class MDSpan<const UnqualifiedValueType, Extents, LayoutPolicy>;
  using ThatClass = MDSpan<DataType, Extents, LayoutPolicy>;
  static constexpr bool IsConst = std::is_const_v<DataType>;

 public:

  using value_type = DataType;
  using ExtentsType = Extents;
  using ExtentIndexType = Extents::ExtentIndexType;
  using LayoutPolicyType = LayoutPolicy;
  using MDIndexType = typename Extents::MDIndexType;
  using LoopIndexType = MDIndexType;
  using ArrayExtentsWithOffsetType = ArrayExtentsWithOffset<Extents, LayoutPolicy>;
  using DynamicDimsType = typename Extents::DynamicDimsType;
  using RemovedFirstExtentsType = typename Extents::RemovedFirstExtentsType;

  // For compatibility. To be removed for consistency with other 'using'
  using ArrayBoundsIndexType = typename Extents::MDIndexType;
  using IndexType = typename Extents::MDIndexType;

 public:

  MDSpan() = default;
  constexpr MDSpan(DataType* ptr, ArrayExtentsWithOffsetType extents)
  : m_ptr(ptr)
  , m_extents(extents)
  {
  }
  constexpr MDSpan(DataType* ptr, const DynamicDimsType& dims)
  : m_ptr(ptr)
  , m_extents(dims)
  {}
  // Constructor MDSpan<const T> from an MDSpan<T>
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr MDSpan(const MDSpan<X, Extents>& rhs)
  : m_ptr(rhs.m_ptr)
  , m_extents(rhs.m_extents)
  {}
  constexpr MDSpan(SmallSpan<DataType> v)
  requires(Extents::isDynamic1D() && !IsConst)
  : m_ptr(v.data())
  , m_extents(DynamicDimsType(v.size()))
  {}
  constexpr MDSpan(SmallSpan<const DataType> v)
  requires(Extents::isDynamic1D() && IsConst)
  : m_ptr(v.data())
  , m_extents(DynamicDimsType(v.size()))
  {}
  constexpr ThatClass& operator=(SmallSpan<DataType> v)
  requires(Extents::isDynamic1D() && !IsConst)
  {
    m_ptr = v.data();
    m_extents = DynamicDimsType(v.size());
    return (*this);
  }
  constexpr ThatClass& operator=(SmallSpan<const DataType> v)
  requires(Extents::isDynamic1D() && IsConst)
  {
    m_ptr = v.data();
    m_extents = DynamicDimsType(v.size());
    return (*this);
  }

 public:

  //! Base pointer for allocated memory
  constexpr DataType* data() { return m_ptr; }
  //! Base pointer for allocated memory
  constexpr const DataType* data() const { return m_ptr; }

 public:

  constexpr DataType* _internalData() { return m_ptr; }
  constexpr const DataType* _internalData() const { return m_ptr; }

 public:

  ArrayExtents<Extents> extents() const
  {
    return m_extents.extents();
  }
  ArrayExtentsWithOffsetType extentsWithOffset() const
  {
    return m_extents;
  }

 public:

  //! Value of the first dimension
  constexpr ExtentIndexType extent0() const requires(Extents::rank() >= 1) { return m_extents.extent0(); }
  //! Value of the second dimension
  constexpr ExtentIndexType extent1() const requires(Extents::rank() >= 2) { return m_extents.extent1(); }
  //! Value of the third dimension
  constexpr ExtentIndexType extent2() const requires(Extents::rank() >= 3) { return m_extents.extent2(); }
  //! Value of the fourth dimension
  constexpr ExtentIndexType extent3() const requires(Extents::rank() >= 4) { return m_extents.extent3(); }

 public:

  //! Value for element \a i,j,k,l
  constexpr Int64 offset(ExtentIndexType i, ExtentIndexType j,
                         ExtentIndexType k, ExtentIndexType l) const
  requires(Extents::rank() == 4)
  {
    return m_extents.offset(i, j, k, l);
  }
  //! Value for element \a i,j,k
  constexpr Int64 offset(ExtentIndexType i, ExtentIndexType j,
                         ExtentIndexType k) const
  requires(Extents::rank() == 3)
  {
    return m_extents.offset(i, j, k);
  }
  //! Value for element \a i,j
  constexpr Int64 offset(ExtentIndexType i, ExtentIndexType j) const
  requires(Extents::rank() == 2)
  {
    return m_extents.offset(i, j);
  }
  //! Value for element \a i
  constexpr Int64 offset(ExtentIndexType i) const
  requires(Extents::rank() == 1) { return m_extents.offset(i); }

  //! Value for element \a idx
  constexpr Int64 offset(MDIndexType idx) const
  {
    return m_extents.offset(idx);
  }

 public:

  //! Value for element \a i,j,k,l
  constexpr DataType& operator()(ExtentIndexType i, ExtentIndexType j,
                                 ExtentIndexType k, ExtentIndexType l) const
  requires(Extents::rank() == 4)
  {
    return m_ptr[offset(i, j, k, l)];
  }
  //! Value for element \a i,j,k
  constexpr DataType& operator()(ExtentIndexType i, ExtentIndexType j, ExtentIndexType k) const
  requires(Extents::rank() == 3)
  {
    return m_ptr[offset(i, j, k)];
  }
  //! Value for element \a i,j
  constexpr DataType& operator()(ExtentIndexType i, ExtentIndexType j) const
  requires(Extents::rank() == 2)
  {
    return m_ptr[offset(i, j)];
  }
  //! Value for element \a i
  constexpr DataType& operator()(ExtentIndexType i) const
  requires(Extents::rank() == 1)
  {
    return m_ptr[offset(i)];
  }
  //! Value for element \a i
  constexpr DataType operator[](ExtentIndexType i) const
  requires(Extents::rank() == 1)
  {
    return m_ptr[offset(i)];
  }

  //! Value for element \a idx
  constexpr DataType& operator()(MDIndexType idx) const
  {
    return m_ptr[offset(idx)];
  }

 public:

  //! Pointer to the value for element \a i,j,k,l
  constexpr DataType* ptrAt(ExtentIndexType i, ExtentIndexType j,
                            ExtentIndexType k, ExtentIndexType l) const
  requires(Extents::rank() == 4)
  {
    return m_ptr + offset(i, j, k, l);
  }
  //! Pointer to the value for element \a i,j,k
  constexpr DataType* ptrAt(ExtentIndexType i, ExtentIndexType j,
                            ExtentIndexType k) const
  requires(Extents::rank() == 3)
  {
    return m_ptr + offset(i, j, k);
  }
  //! Pointer to the value for element \a i,j
  constexpr DataType* ptrAt(ExtentIndexType i, ExtentIndexType j) const
  requires(Extents::rank() == 2)
  {
    return m_ptr + offset(i, j);
  }
  //! Pointer to the value for element \a i
  constexpr DataType* ptrAt(ExtentIndexType i) const
  requires(Extents::rank() == 1)
  {
    return m_ptr + offset(i);
  }

  //! Pointer to the value for element \a i
  constexpr DataType* ptrAt(MDIndexType idx) const
  {
    return m_ptr + offset(idx);
  }

 public:

  /*!
   * \brief Returns a dimension (N-1) view starting from index element \a i.
   *
   * For example:
   * \code
   *   MDSpan<Real, MDDim3> span3 = ...;
   *   MDSpan<Real, MDDim2> sliced_span = span3.slice(5);
   *   // sliced_span(i,i) <=> span3(5,i,j);
   * \endcode
   *
   * \warning This is only valid if \a LayoutPolicy is \a RightLayout.
   */
  ARCCORE_HOST_DEVICE MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>
  slice(ExtentIndexType i) const requires(Extents::rank() >= 2 && std::is_base_of_v<RightLayout, LayoutPolicy>)
  {
    auto new_extents = m_extents.extents().removeFirstExtent().dynamicExtents();
    std::array<ExtentIndexType, ExtentsType::rank()> indexes = {};
    indexes[0] = i;
    DataType* base_ptr = this->ptrAt(MDIndexType(indexes));
    return MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>(base_ptr, new_extents);
  }

 public:

  constexpr MDSpan<const DataType, Extents, LayoutPolicy> constSpan() const
  {
    return MDSpan<const DataType, Extents, LayoutPolicy>(m_ptr, m_extents);
  }

  constexpr MDSpan<const DataType, Extents, LayoutPolicy> constMDSpan() const
  {
    return MDSpan<const DataType, Extents, LayoutPolicy>(m_ptr, m_extents);
  }

  constexpr Span<DataType> to1DSpan() const
  {
    return { m_ptr, m_extents.totalNbElement() };
  }

  constexpr SmallSpan<DataType> to1DSmallSpan() requires(Extents::rank() == 1)
  {
    // TODO: maybe add a test in Check mode to make sure extent0() fit in a 'Int32'
    return { _internalData(), static_cast<Int32>(extent0()) };
  }
  constexpr SmallSpan<const DataType> to1DSmallSpan() const requires(Extents::rank() == 1)
  {
    return to1DConstSmallSpan();
  }
  constexpr SmallSpan<const DataType> to1DConstSmallSpan() const requires(Extents::rank() == 1)
  {
    // TODO: maybe add a test in Check mode to make sure extent0() fit in a 'Int32'
    return { _internalData(), static_cast<Int32>(extent0()) };
  }

 private:

  DataType* m_ptr = nullptr;
  ArrayExtentsWithOffsetType m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
