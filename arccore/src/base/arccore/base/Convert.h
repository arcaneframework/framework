// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2026 */
/*                                                                           */
/* Functions to convert a character string into a given type.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CONVERT_H
#define ARCCORE_BASE_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#ifndef ARCCORE_COMPILING_FRAMEWORK
// This header is not needed and will be removed in July 2027
#include <iostream>
#endif
#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Real to \c double
inline double
toDouble(Real r)
{
#ifdef ARCCORE_REAL_USE_APFLOAT
  return ap2double(r.ap);
#else
  return static_cast<double>(r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Real to \c Integer
inline Integer
toInteger(Real r)
{
  return static_cast<Integer>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Real to \c Int64
inline Int64
toInt64(Real r)
{
  return static_cast<Int64>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Real to \c Int32
inline Int32
toInt32(Real r)
{
  return static_cast<Int32>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converts a \c Real to \c Integer
inline bool
toBool(Real r)
{
  return static_cast<bool>(toDouble(r));
}

//! Converts \c r to a \c Real
inline Real
toReal(Real r)
{
  return r;
}

//! Converts \c r to a \c Real
inline Real
toReal(int r)
{
  return static_cast<Real>(r);
}

//! Converts \c r to a \c Real
inline Real
toReal(unsigned int r)
{
  return static_cast<Real>(r);
}

//! Converts \c r to a \c Real
inline Real
toReal(long r)
{
  return static_cast<Real>(r);
}

//! Converts \c r to a \c Real
inline Real
toReal(unsigned long r)
{
  return static_cast<Real>(r);
}

//! Converts \c r to a \c Real
inline Real
toReal(long long r)
{
#ifdef ARCCORE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<long>(r));
#else
  return static_cast<Real>(r);
#endif
}

//! Converts \c r to a \c Real
inline Real
toReal(unsigned long long r)
{
#ifdef ARCCORE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<unsigned long>(r));
#else
  return static_cast<Real>(r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Template class for converting a type.
 *
 * Currently, this is only available via a specialization
 * for the types 'Int32', 'Int64', and 'Real'.
 */
template <typename T>
class Type;

template <typename T>
class ScalarType
{
 public:

  //! Converts \a s to type \a T
  ARCCORE_BASE_EXPORT static std::optional<T> tryParse(StringView s);

  /*!
   * \brief Converts \a s to type \a T.
   *
   * If \a s.empty() is true, then it returns \a default_value.
   */
  static std::optional<T>
  tryParseIfNotEmpty(StringView s, const T& default_value)
  {
    return (s.empty()) ? default_value : tryParse(s);
  }

  /*!
   * \brief Converts the value of the environment variable \a s to type \a T.
   *
   * If platform::getEnvironmentVariable(s) is null, return std::nullopt.
   * Otherwise, it returns this value converted to type \a T. If the conversion
   * is not possible, it returns std::nullopt if \a throw_if_invalid is \a false or
   * throws an exception if it is \a true.
   */
  ARCCORE_BASE_EXPORT static std::optional<T>
  tryParseFromEnvironment(StringView s, bool throw_if_invalid);
};

//! Specialization for scalar types
template <> class Type<Int64> : public ScalarType<Int64>
{};
template <> class Type<Int32> : public ScalarType<Int32>
{};
template <> class Type<Real> : public ScalarType<Real>
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
