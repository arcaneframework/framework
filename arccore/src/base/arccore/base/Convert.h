// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Functions to convert a character string into a given type.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CONVERT_H
#define ARCCORE_BASE_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#include <iostream>
#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{

/*!
 * \brief Encapsulates an std::istream for a StringView.
 *
 * Currently (C++20) std::istringstream uses an
 * input std::string, which requires an instance of this type
 * and thus a potential allocation. This class serves to avoid
 * this by directly using the memory pointed to by the instance
 * of StringView passed in the constructor. The latter must
 * remain valid throughout the use of this class.
 */
class ARCCORE_BASE_EXPORT StringViewInputStream
: private std::streambuf
{
 public:

  explicit StringViewInputStream(StringView v);

 public:

  std::istream& stream() { return m_stream; }

 private:

  StringView m_view;
  std::istream m_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

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
 * for the types 'Int32', 'Int64', and 'Real3'.
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
