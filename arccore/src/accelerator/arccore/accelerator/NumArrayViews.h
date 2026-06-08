// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.h                                             (C) 2000-2025 */
/*                                                                           */
/* Management of views for 'NumArray' for accelerators.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_NUMARRAYVIEWS_H
#define ARCCORE_ACCELERATOR_NUMARRAYVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/NumArray.h"
#include "arccore/common/DataView.h"
#include "arccore/common/accelerator/ViewBuildInfo.h"

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extents, typename LayoutPolicy>
class NumArrayViewSetter;
template <typename Accessor, typename Extents, typename LayoutPolicy>
class NumArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for views on 'NumArray'.
 */
class ARCCORE_ACCELERATOR_EXPORT NumArrayViewBase
{
 protected:

  // Does not use \a command yet
  // but it should not be deleted
  explicit NumArrayViewBase(const ViewBuildInfo&, Span<const std::byte> bytes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read, write, or read/write view on a 'NumArray'.
 *
 * Views work up to rank 4 arrays.
 */
template <typename Accessor, typename Extents, typename LayoutType>
class NumArrayView
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType, Extents, LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(const ViewBuildInfo& command, SpanType v)
  : NumArrayViewBase(command, Arccore::asBytes(v.to1DSpan()))
  , m_values(v)
  {}

  //! Accessor for a rank 1 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  //! Accessor for a rank 1 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }
  //! Accessor for a rank 1 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  //! Accessor for a rank 1 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](ArrayIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accessor for a rank 2 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j) const
  {
    return Accessor::build(m_values.ptrAt(i, j));
  }
  //! Accessor for a rank 2 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<2> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accessor for a rank 3 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j, Int32 k) const
  {
    return Accessor::build(m_values.ptrAt(i, j, k));
  }
  //! Accessor for a rank 3 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<3> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accessor for a rank 4 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 4, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j, Int32 k, Int32 l) const
  {
    return Accessor::build(m_values.ptrAt(i, j, k, l));
  }
  //! Accessor for a rank 4 array
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 4, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<4> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Converted to a 1D view.
  constexpr ARCCORE_HOST_DEVICE Span<DataType> to1DSpan() const
  {
    return m_values.to1DSpan();
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Write view.
 */
template <typename DataType, typename Extents, typename LayoutPolicy> auto
viewOut(const ViewBuildInfo& command, NumArray<DataType, Extents, LayoutPolicy>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutPolicy>(command, var.mdspan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read/write view.
 */
template <typename DataType, typename Extents, typename LayoutPolicy> auto
viewInOut(const ViewBuildInfo& command, NumArray<DataType, Extents, LayoutPolicy>& v)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutPolicy>(command, v.mdspan());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view.
 */
template <typename DataType, typename Extents, typename LayoutType> auto
viewIn(const ViewBuildInfo& command, const NumArray<DataType, Extents, LayoutType>& v)
{
  using Accessor = DataViewGetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutType>(command, v.constMDSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Input view on a NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayInView = NumArrayView<DataViewGetter<DataType>, Extents, LayoutType>;

//! Output view on a NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayOutView = NumArrayView<DataViewSetter<DataType>, Extents, LayoutType>;

//! Input/output view on a NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayInOutView = NumArrayView<DataViewGetterSetter<DataType>, Extents, LayoutType>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
