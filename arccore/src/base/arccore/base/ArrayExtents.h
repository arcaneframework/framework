// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExtents.h                                              (C) 2000-2026 */
/*                                                                           */
/* Management of the number of elements per dimension for N-dimensional      */
/* arrays.                                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYEXTENTS_H
#define ARCCORE_BASE_ARRAYEXTENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/MDIndex.h"
#include "arccore/base/ArrayLayout.h"
#include "arccore/base/ArrayExtentsValue.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization of ArrayStrideBase for 0-dimensional arrays (scalars)
 */
template <>
class ArrayStridesBase<0>
{
 public:

  ArrayStridesBase() = default;
  //! Value of the stride for the i-th dimension.
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return {}; }
  //! Value total of the stride
  ARCCORE_HOST_DEVICE Int64 totalStride() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayStridesBase<0> fromSpan([[maybe_unused]] Span<const Int32> strides)
  {
    // TODO: check the size of \a strides
    return {};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to maintain the stride in each dimension.
 *
 * The stride for a dimension is the memory distance between two elements
 * of the array for that dimension. Generally, the stride is equal to the number
 * of elements in the dimension unless padding is used,
 * for example, to align certain dimensions.
 */
template <int RankValue>
class ArrayStridesBase
{
 public:

  ArrayStridesBase() = default;
  //! Value of the stride for the i-th dimension.
  ARCCORE_HOST_DEVICE Int32 stride(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE Int32 operator()(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return { m_strides.data(), RankValue }; }
  //! Total stride value
  ARCCORE_HOST_DEVICE Int64 totalStride() const
  {
    Int64 nb_element = 1;
    for (int i = 0; i < RankValue; i++)
      nb_element *= m_strides[i];
    return nb_element;
  }
  // Instance containing dimensions after the first
  ARCCORE_HOST_DEVICE ArrayStridesBase<RankValue - 1> removeFirstStride() const
  {
    return ArrayStridesBase<RankValue - 1>::fromSpan({ m_strides.data() + 1, RankValue - 1 });
  }
  /*!
   * \brief Constructs an instance from the values given in \a stride.
   * \pre stride.size() == RankValue.
   */
  ARCCORE_HOST_DEVICE static ArrayStridesBase<RankValue> fromSpan(Span<const Int32> strides)
  {
    ArrayStridesBase<RankValue> v;
    // TODO: check the size
    for (int i = 0; i < RankValue; ++i)
      v.m_strides[i] = strides[i];
    return v;
  }

 protected:

  std::array<Int32, RankValue> m_strides = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization of ArrayExtentsBase for 0-dimensional arrays (scalars)
 */
template <>
class ArrayExtentsBase<ExtentsV<>>
{
 public:

  ArrayExtentsBase() = default;
  //! Number of elements in the i-th dimension.
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return {}; }
  //! Total number of elements
  constexpr ARCCORE_HOST_DEVICE Int32 totalNbElement() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayExtentsBase<ExtentsV<>> fromSpan([[maybe_unused]] Span<const Int32> extents)
  {
    // TODO: check the size of \a extents
    return {};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class to maintain the number of elements in each dimension.
 */
template <typename Extents>
class ArrayExtentsBase
: protected Extents::ArrayExtentsValueType
{
 protected:

  using BaseClass = Extents::ArrayExtentsValueType;
  using ExtentIndexType = Extents::ExtentIndexType;
  using ArrayExtentsPreviousRank = ArrayExtentsBase<typename Extents::RemovedFirstExtentsType>;
  using DynamicDimsType = Extents::DynamicDimsType;

 public:

  using BaseClass::asStdArray;
  using BaseClass::constExtent;
  using BaseClass::dynamicExtents;
  using BaseClass::getIndices;
  using BaseClass::totalNbElement;

 public:

  ARCCORE_HOST_DEVICE constexpr ArrayExtentsBase()
  : BaseClass()
  {}

 protected:

  explicit constexpr ARCCORE_HOST_DEVICE ArrayExtentsBase(SmallSpan<const ExtentIndexType> extents)
  : BaseClass(extents)
  {
  }

  explicit constexpr ARCCORE_HOST_DEVICE ArrayExtentsBase(DynamicDimsType extents)
  : BaseClass(extents)
  {
  }

 public:

  //! TEMPORARY: Sets the number of elements of dimension 0 to \a v.
  ARCCORE_HOST_DEVICE void setExtent0(ExtentIndexType v) { this->m_extent0.v = v; }

  // Instance containing dimensions after the first
  ARCCORE_HOST_DEVICE ArrayExtentsPreviousRank removeFirstExtent() const
  {
    auto x = BaseClass::_removeFirstExtent();
    return ArrayExtentsPreviousRank::fromSpan(x);
  }

  // Number of elements of the I-th dimension converted to an 'Int64'.
  template <Int32 I> constexpr ARCCORE_HOST_DEVICE Int64 constLargeExtent() const
  {
    return BaseClass::template constExtent<I>();
  }

  /*!
   * \brief Constructs an instance from the values given in \a extents.
   */
  ARCCORE_HOST_DEVICE static ArrayExtentsBase fromSpan(SmallSpan<const ExtentIndexType> extents)
  {
    return ArrayExtentsBase(extents);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent for 1-dimensional arrays.
 */
template <typename SizeType_, Int32 X0>
class ArrayExtents<ExtentsV<SizeType_, X0>>
: public ArrayExtentsBase<ExtentsV<SizeType_, X0>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0>;
  using BaseClass = ArrayExtentsBase<ExtentsType>;
  using BaseClass::extent0;
  using BaseClass::totalNbElement;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  ArrayExtents() = default;
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const BaseClass& rhs)
  : BaseClass(rhs)
  {}
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const DynamicDimsType& extents)
  : BaseClass(extents)
  {
  }
  // TODO: To be removed
  constexpr ARCCORE_HOST_DEVICE explicit ArrayExtents(ExtentIndexType dim1_size)
  {
    static_assert(ExtentsType::nb_dynamic == 1, "This method is only allowed for full dynamic extents");
    this->m_extent0.v = dim1_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent for 2-dimensional arrays.
 */
template <typename SizeType_, Int32 X0, Int32 X1>
class ArrayExtents<ExtentsV<SizeType_, X0, X1>>
: public ArrayExtentsBase<ExtentsV<SizeType_, X0, X1>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1>;
  using BaseClass = ArrayExtentsBase<ExtentsType>;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::totalNbElement;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  ArrayExtents() = default;
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const BaseClass& rhs)
  : BaseClass(rhs)
  {}
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const DynamicDimsType& extents)
  : BaseClass(extents)
  {
  }
  // TODO: To be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(ExtentIndexType dim1_size, ExtentIndexType dim2_size)
  {
    static_assert(ExtentsType::nb_dynamic == 2, "This method is only allowed for full dynamic extents");
    this->m_extent0.v = dim1_size;
    this->m_extent1.v = dim2_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent for 3-dimensional arrays.
 */
template <typename SizeType_, Int32 X0, Int32 X1, Int32 X2>
class ArrayExtents<ExtentsV<SizeType_, X0, X1, X2>>
: public ArrayExtentsBase<ExtentsV<SizeType_, X0, X1, X2>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1, X2>;
  using BaseClass = ArrayExtentsBase<ExtentsType>;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::extent2;
  using BaseClass::totalNbElement;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  ArrayExtents() = default;
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const BaseClass& rhs)
  : BaseClass(rhs)
  {}
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const DynamicDimsType& extents)
  : BaseClass(extents)
  {
  }
  // TODO: To be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(ExtentIndexType dim1_size, ExtentIndexType dim2_size, ExtentIndexType dim3_size)
  {
    static_assert(ExtentsType::nb_dynamic == 3, "This method is only allowed for full dynamic extents");
    this->m_extent0.v = dim1_size;
    this->m_extent1.v = dim2_size;
    this->m_extent2.v = dim3_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent for 4-dimensional arrays.
 */
template <typename SizeType_, Int32 X0, Int32 X1, Int32 X2, Int32 X3>
class ArrayExtents<ExtentsV<SizeType_, X0, X1, X2, X3>>
: public ArrayExtentsBase<ExtentsV<SizeType_, X0, X1, X2, X3>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1, X2, X3>;
  using BaseClass = ArrayExtentsBase<ExtentsType>;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::extent2;
  using BaseClass::extent3;
  using BaseClass::totalNbElement;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  ArrayExtents() = default;
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const BaseClass& rhs)
  : BaseClass(rhs)
  {}
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(const DynamicDimsType& extents)
  : BaseClass(extents)
  {
  }
  // TODO: To be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtents(ExtentIndexType dim1_size, ExtentIndexType dim2_size, ExtentIndexType dim3_size, ExtentIndexType dim4_size)
  {
    static_assert(ExtentsType::nb_dynamic == 4, "This method is only allowed for full dynamic extents");
    this->m_extent0.v = dim1_size;
    this->m_extent1.v = dim2_size;
    this->m_extent2.v = dim3_size;
    this->m_extent3.v = dim4_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent and Offset for 1-dimensional arrays.
 */
template <typename SizeType_, SizeType_ X0, typename LayoutType>
class ArrayExtentsWithOffset<ExtentsV<SizeType_, X0>, LayoutType>
: private ArrayExtents<ExtentsV<SizeType_, X0>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0>;
  using BaseClass = ArrayExtents<ExtentsType>;
  using BaseClass::asStdArray;
  using BaseClass::extent0;
  using BaseClass::getIndices;
  using BaseClass::totalNbElement;
  using Layout = typename LayoutType::Layout1Type;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = MDIndexType;

 public:

  ArrayExtentsWithOffset() = default;
  // TODO: to be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(const ArrayExtents<ExtentsType>& rhs)
  : BaseClass(rhs)
  {
  }
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(const DynamicDimsType& rhs)
  : BaseClass(rhs)
  {
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(ExtentIndexType i) const
  {
    BaseClass::_checkIndex(i);
    return i;
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(MDIndexType idx) const
  {
    BaseClass::_checkIndex(idx.id0());
    return idx.id0();
  }
  constexpr BaseClass extents() const
  {
    const BaseClass* b = this;
    return *b;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent and Offset for 2-dimensional arrays.
 */
template <typename SizeType_, SizeType_ X0, SizeType_ X1, typename LayoutType>
class ArrayExtentsWithOffset<ExtentsV<SizeType_, X0, X1>, LayoutType>
: private ArrayExtents<ExtentsV<SizeType_, X0, X1>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1>;
  using BaseClass = ArrayExtents<ExtentsType>;
  using BaseClass::asStdArray;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::getIndices;
  using BaseClass::totalNbElement;
  using Layout = typename LayoutType::Layout2Type;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = MDIndexType;

 public:

  ArrayExtentsWithOffset() = default;
  // TODO: to be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<ExtentsType> rhs)
  : BaseClass(rhs)
  {
  }
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(const DynamicDimsType& rhs)
  : BaseClass(rhs)
  {
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(ExtentIndexType i, ExtentIndexType j) const
  {
    return offset({ i, j });
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(MDIndexType idx) const
  {
    BaseClass::_checkIndex(idx);
    return Layout::offset(idx, this->template constExtent<Layout::LastExtent>());
  }
  constexpr BaseClass extents() const
  {
    const BaseClass* b = this;
    return *b;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Extent and Offset for 3-dimensional arrays.
 */
template <typename SizeType_, SizeType_ X0, SizeType_ X1, SizeType_ X2, typename LayoutType>
class ArrayExtentsWithOffset<ExtentsV<SizeType_, X0, X1, X2>, LayoutType>
: private ArrayExtents<ExtentsV<SizeType_, X0, X1, X2>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1, X2>;
  using BaseClass = ArrayExtents<ExtentsType>;
  using BaseClass::asStdArray;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::extent2;
  using BaseClass::getIndices;
  using BaseClass::totalNbElement;
  using Layout = typename LayoutType::Layout3Type;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = typename BaseClass::MDIndexType;

 public:

  ArrayExtentsWithOffset() = default;
  // TODO: to be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<ExtentsType> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(const DynamicDimsType& rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(ExtentIndexType i, ExtentIndexType j, ExtentIndexType k) const
  {
    return offset({ i, j, k });
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(MDIndexType idx) const
  {
    this->_checkIndex(idx);
    return Layout::offset(idx, this->template constExtent<Layout::LastExtent>(), m_dim23_size);
  }
  constexpr BaseClass extents() const
  {
    const BaseClass* b = this;
    return *b;
  }

 protected:

  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    const BaseClass& b = *this;
    m_dim23_size = Layout::computeOffsetIndexes(b);
  }

 private:

  Int64 m_dim23_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extent and Offset for 4-dimensional arrays.
 */
template <typename SizeType_, SizeType_ X0, SizeType_ X1, SizeType_ X2, SizeType_ X3, typename LayoutType>
class ArrayExtentsWithOffset<ExtentsV<SizeType_, X0, X1, X2, X3>, LayoutType>
: private ArrayExtents<ExtentsV<SizeType_, X0, X1, X2, X3>>
{
 public:

  using ExtentsType = ExtentsV<SizeType_, X0, X1, X2, X3>;
  using BaseClass = ArrayExtents<ExtentsType>;
  using BaseClass::asStdArray;
  using BaseClass::extent0;
  using BaseClass::extent1;
  using BaseClass::extent2;
  using BaseClass::extent3;
  using BaseClass::getIndices;
  using BaseClass::totalNbElement;
  using Layout = typename LayoutType::Layout4Type;
  using DynamicDimsType = BaseClass::DynamicDimsType;
  using MDIndexType = BaseClass::MDIndexType;
  using ExtentIndexType = BaseClass::ExtentIndexType;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = typename BaseClass::MDIndexType;

 public:

  ArrayExtentsWithOffset() = default;
  // TODO: to be removed
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<ExtentsType> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  constexpr ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(const DynamicDimsType& rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(ExtentIndexType i, ExtentIndexType j, ExtentIndexType k, ExtentIndexType l) const
  {
    return offset({ i, j, k, l });
  }
  constexpr ARCCORE_HOST_DEVICE Int64 offset(MDIndexType idx) const
  {
    this->_checkIndex(idx);
    return (m_dim234_size * idx.largeId0()) + m_dim34_size * idx.largeId1() + this->m_extent3.v * idx.largeId2() + idx.largeId3();
  }
  BaseClass extents() const
  {
    const BaseClass* b = this;
    return *b;
  }

 protected:

  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    m_dim34_size = Int64(this->m_extent2.v) * Int64(this->m_extent3.v);
    m_dim234_size = Int64(m_dim34_size) * Int64(this->m_extent1.v);
  }

 private:

  Int64 m_dim34_size = 0; //!< dim3 * dim4
  Int64 m_dim234_size = 0; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
