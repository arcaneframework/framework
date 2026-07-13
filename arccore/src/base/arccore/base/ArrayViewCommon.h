// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayViewCommon.h                                           (C) 2000-2026 */
/*                                                                           */
/* Common declarations for the ArrayView, ConstArrayView, and Span classes.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEWCOMMON_H
#define ARCCORE_BASE_ARRAYVIEWCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayIterator.h"

// This header file is only required when we want to print values of the view.
// We can remove it but in this case all user code has to explicitly include
// it so it may break source compatibility. So at the moment (July 2026)
// we include it.
#include "arccore/base/ArrayViewDumper.h"

#include <iosfwd>
// 'assert' is necessary for accelerator code
#include <assert.h>

#ifndef ARCCORE_COMPILING_FRAMEWORK
// This header is not needed and will be removed in July 2027
#include <iostream>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ViewType>
class ArrayViewDumper;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Sub-view corresponding to the interval `index` over `nb_interval`
template <typename ViewType> ARCCORE_HOST_DEVICE auto
subViewInterval(ViewType view,
                typename ViewType::size_type index,
                typename ViewType::size_type nb_interval) -> ViewType
{
  using size_type = typename ViewType::size_type;
  if (nb_interval <= 0)
    return ViewType();
  if (index < 0 || index >= nb_interval)
    return ViewType();
  size_type n = view.size();
  size_type isize = n / nb_interval;
  size_type ibegin = index * isize;
  // For the last interval, take the remaining elements
  if ((index + 1) == nb_interval)
    isize = n - ibegin;
  return ViewType::create(view.data() + ibegin, isize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Displays the values of the view.
 *
 * Displays the values of `val` on the stream `o`.
 * If `max_print` is greater than 0, it indicates the maximum number of values
 * to display.
 */
template <typename ViewType> inline void
dumpArray(std::ostream& o, ViewType val, int max_print)
{
  ArrayViewDumper<ViewType>::dumpArray(o, val, max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if the two views are equal
template <typename ViewType> inline bool
areEqual(ViewType rhs, ViewType lhs)
{
  using size_type = typename ViewType::size_type;
  if (rhs.size() != lhs.size())
    return false;
  size_type s = rhs.size();
  for (size_type i = 0; i < s; ++i) {
    if (rhs[i] != lhs[i])
      return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if the two views are equal
template <typename View2DType> inline bool
areEqual2D(View2DType rhs, View2DType lhs)
{
  using size_type = typename View2DType::size_type;
  const size_type dim1_size = rhs.dim1Size();
  const size_type dim2_size = rhs.dim2Size();
  if (dim1_size != lhs.dim1Size())
    return false;
  if (dim2_size != lhs.dim2Size())
    return false;
  for (size_type i = 0; i < dim1_size; ++i) {
    for (size_type j = 0; j < dim2_size; ++j) {
      if (rhs(i, j) != lhs(i, j))
        return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Throws an 'ArgumentException'
extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowTooBigInteger [[noreturn]] (std::size_t size);

extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowTooBigInt64 [[noreturn]] (std::size_t size);

extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowNegativeSize [[noreturn]] (Int64 size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tests if `size` is positive or zero and throws an exception otherwise
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsPositive(Int64 size)
{
  if (size < 0) {
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is negative");
#else
    Arcane::Impl::arccoreThrowNegativeSize(size);
#endif
  }
}

//! Tests if `size` is smaller than ARCCORE_INTEGER_MAX and throws an exception otherwise
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsValidInteger(Int64 size)
{
  if (size >= ARCCORE_INTEGER_MAX) {
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is bigger than ARCCORE_INTEGER_MAX");
#else
    Arcane::Impl::arccoreThrowTooBigInteger(size);
#endif
  }
}

//! Tests if `size` is smaller than ARCCORE_INT64_MAX and throws an exception otherwise
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsValidInt64(size_t size)
{
  if (size >= ARCCORE_INT64_MAX) {
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is bigger than ARCCORE_INT64_MAX");
#else
    Arcane::Impl::arccoreThrowTooBigInt64(size);
#endif
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(unsigned long long size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(long long size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  Arcane::Impl::arccoreCheckIsPositive(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 *
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(long size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  Arcane::Impl::arccoreCheckIsPositive(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(unsigned int size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Integer' to serve
 * as an array size.
 * If possible, returns `size` converted to an 'Integer'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(int size)
{
  Arcane::Impl::arccoreCheckIsValidInteger(size);
  Arcane::Impl::arccoreCheckIsPositive(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Checks that `size` can be converted into an 'Int64' to serve
 * as an array size.
 *
 * If possible, returns `size` converted to an 'Int64'. Otherwise, throws
 * an ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Int64
arccoreCheckLargeArraySize(size_t size)
{
  Arcane::Impl::arccoreCheckIsValidInt64(size);
  return static_cast<Int64>(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IntType> class ArraySizeChecker;

//! Specialization to test conversion to Int32
template <>
class ArraySizeChecker<Int32>
{
 public:

  template <typename SizeType> ARCCORE_HOST_DEVICE static Int32 check(SizeType size)
  {
    return arccoreCheckArraySize(size);
  }
};

//! Specialization to test conversion to Int64
template <>
class ArraySizeChecker<Int64>
{
 public:

  static ARCCORE_HOST_DEVICE Int64 check(std::size_t size)
  {
    return arccoreCheckLargeArraySize(size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::impl
{
using Arcane::Impl::arccoreCheckIsPositive;
using Arcane::Impl::arccoreCheckIsValidInt64;
using Arcane::Impl::arccoreCheckIsValidInteger;
using Arcane::Impl::arccoreThrowNegativeSize;
using Arcane::Impl::arccoreThrowTooBigInt64;
using Arcane::Impl::arccoreThrowTooBigInteger;
using Arcane::Impl::areEqual;
using Arcane::Impl::areEqual2D;
using Arcane::Impl::dumpArray;
using Arcane::Impl::subViewInterval;
} // namespace Arccore::impl

namespace Arccore
{
using Arcane::arccoreCheckArraySize;
using Arcane::arccoreCheckLargeArraySize;
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
