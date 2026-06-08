// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckedConvert.h                                            (C) 2000-2025 */
/*                                                                           */
/* Functions to convert one type to another with checking.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CHECKEDCONVERT_H
#define ARCCORE_BASE_CHECKEDCONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BadCastException.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/Convert.h"

#include <limits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Functions to convert one type to another with checking
 */
namespace Arcane::CheckedConvert::Impl
{
  inline Integer
  toInteger(Int64 v)
  {
    if (v > std::numeric_limits<Integer>::max() || v < std::numeric_limits<Integer>::min())
      ARCCORE_THROW(BadCastException, "Invalid conversion from '{0}' to type Integer", v);
    return static_cast<Integer>(v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::CheckedConvert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Int64 to an \c Integer
inline Integer
toInteger(Real r)
{
  double v = Convert::toDouble(r);
  if (v > static_cast<double>(std::numeric_limits<Integer>::max()) || v < static_cast<double>(std::numeric_limits<Integer>::min()))
    ARCCORE_THROW(BadCastException, "Invalid conversion from '{0}' to type Integer", v);
  return static_cast<Integer>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts \a v to the \c Integer type
inline Integer
toInteger(long long v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

//! Converts \a v to the \c Integer type
inline Integer
toInteger(long v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

//! Converts \a v to the \c Integer type
inline Integer
toInteger(int v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

//! Converts \a v to the \c Integer type
inline Integer
toInteger(unsigned long long v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

//! Converts \a v to the \c Integer type
inline Integer
toInteger(unsigned long v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

//! Converts \a v to the \c Integer type
inline Integer
toInteger(unsigned int v)
{
  return Impl::toInteger(static_cast<Int64>(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Int64 to a \c Int32
inline Int32
toInt32(Int64 v)
{
  if (v > std::numeric_limits<Int32>::max() || v < std::numeric_limits<Int32>::min())
    ARCCORE_THROW(BadCastException,"Invalid conversion from '{0}' to type Int32", v);
  return static_cast<Int32>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Int64 to a \c Int16
inline Int16
toInt16(Int64 v)
{
  if (v > std::numeric_limits<Int16>::max() || v < std::numeric_limits<Int16>::min())
    ARCCORE_THROW(BadCastException,"Invalid conversion from '{0}' to type Int16", v);
  return static_cast<Int16>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplies three 'Integer's and checks that the result can be contained
 * in an 'Integer'.
 */
inline Integer
multiply(Integer x, Integer y, Integer z)
{
  auto x2 = static_cast<Int64>(x);
  auto y2 = static_cast<Int64>(y);
  auto z2 = static_cast<Int64>(z);
  Int64 xyz = x2 * y2 * z2;
  return Impl::toInteger(xyz);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplies two 'Integer's and checks that the result can be contained
 * in an 'Integer'.
 */
inline Integer
multiply(Integer x, Integer y)
{
  auto x2 = static_cast<Int64>(x);
  auto y2 = static_cast<Int64>(y);
  Int64 xy = x2 * y2;
  return Impl::toInteger(xy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CheckedConvert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
