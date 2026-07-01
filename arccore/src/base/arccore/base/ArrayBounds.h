// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayBounds.h                                               (C) 2000-2026 */
/*                                                                           */
/* Handling of iterations on N-dimensional arrays                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYBOUNDS_H
#define ARCCORE_BASE_ARRAYBOUNDS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayExtents.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Base class for bounds of multidimensional array.
 */
template <typename Extents>
class ArrayBoundsBase
: private ArrayExtents<Extents>
{
 public:

  using BaseClass = ArrayExtents<Extents>;
  using BaseClass::asStdArray;
  using BaseClass::constExtent;
  using BaseClass::getIndices;
  using BaseClass::dynamicExtents;
  using MDIndexType = BaseClass::MDIndexType;
  using LoopIndexType = BaseClass::LoopIndexType;
  using ExtentIndexType = Extents::ExtentIndexType;
  using ArrayExtentType = ArrayExtents<Extents>;

  using IndexType ARCCORE_DEPRECATED_REASON("Y2025: Use 'MDIndexType' instead") = LoopIndexType;

 public:

  ArrayBoundsBase() = default;
  constexpr explicit ArrayBoundsBase(const BaseClass& rhs)
  : ArrayExtents<Extents>(rhs)
  {
  }

  constexpr explicit ArrayBoundsBase(const std::array<ExtentIndexType, Extents::nb_dynamic>& v)
  : BaseClass(v)
  {
  }

 public:

  constexpr Int64 nbElement() const { return this->totalNbElement(); }

 protected:

  using BaseClass::asOtherStdArray;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Represents the bounds of a multidimensional array.
 */
template <typename Extents>
class ArrayBounds
: public ArrayBoundsBase<Extents>
{
  template <typename OtherExtents> friend class ArrayBounds;

 public:

  using ExtentsType = Extents;
  using ExtentIndexType = Extents::ExtentIndexType;
  using BaseClass = ArrayBoundsBase<ExtentsType>;
  using ArrayExtentsType = ArrayExtents<ExtentsType>;
  using LoopIndexType = BaseClass::LoopIndexType;

 public:

  constexpr ArrayBounds(ExtentIndexType dim1, ExtentIndexType dim2,
                        ExtentIndexType dim3, ExtentIndexType dim4) requires(Extents::nb_dynamic == 4)
  : BaseClass(ArrayExtentsType(dim1, dim2, dim3, dim4))
  {
  }

  constexpr ArrayBounds(ExtentIndexType dim1, ExtentIndexType dim2,
                        ExtentIndexType dim3) requires(Extents::nb_dynamic == 3)
  : BaseClass(ArrayExtentsType(dim1, dim2, dim3))
  {
  }

  constexpr ArrayBounds(ExtentIndexType dim1, ExtentIndexType dim2) requires(Extents::nb_dynamic == 2)
  : BaseClass(ArrayExtentsType(dim1, dim2))
  {
  }

  constexpr ArrayBounds(ExtentIndexType dim1) requires(Extents::nb_dynamic == 1)
  : BaseClass(ArrayExtentsType(dim1))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtentsType& v)
  : BaseClass(v)
  {
  }

  constexpr explicit ArrayBounds(std::array<ExtentIndexType, Extents::nb_dynamic>& v)
  : BaseClass(v)
  {
  }

  //! Convert to a ArrayBound with same Extent buf with a different index type
  template <typename OtherArrayBounds> static constexpr ArrayBounds
  fromOther(const OtherArrayBounds& rhs)
  {
    std::array<ExtentIndexType, Extents::nb_dynamic> x = rhs.template asOtherStdArray<ExtentIndexType>();
    return ArrayBounds(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
