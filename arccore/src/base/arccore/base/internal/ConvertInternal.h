// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConvertInternal.h                                           (C) 2000-2026 */
/*                                                                           */
/* Functions to convert a character string into a given type.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_CONVERTINTERNAL_H
#define ARCCORE_BASE_INTERNAL_CONVERTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Information on the behavior of conversion methods
class ARCCORE_BASE_EXPORT ConvertPolicy
{
 public:

  /*!
   * \brief Indicates whether 'std::from_chars' is used to convert
   * character strings into a numeric type.
   *
   * If 'std::from_chars' is not used, functions such as strtod(),
   * strtol(), ... are used.
   *
   * The default in C++20 is to use std::from_chars().
   */
  static void setUseFromChars(bool v) { m_use_from_chars = v; }
  static bool isUseFromChars() { return m_use_from_chars; }

  //! Sets the verbosity level for conversion functions.
  static void setVerbosity(Int32 v) { m_verbosity = v; }
  static bool verbosity() { return m_verbosity; }

  /*!
   * If true, the same mechanism is used to read 'RealN' as to read 'Real'.
   *
   * Before version 3.15 of Arcane, reading 'Real' was done via std::strtod()
   * and reading 'RealN' via std::istream. If \a v is true, std::strtod() is used
   * for everyone (or std::from_chars()) if available.
   */
  static void setUseSameConvertForAllReal(bool v)
  {
    m_use_same_convert_for_all_real = v;
  }
  static bool isUseSameConvertForAllReal()
  {
    return m_use_same_convert_for_all_real;
  }

 private:

  static Int32 m_verbosity;
  static bool m_use_from_chars;
  static bool m_use_same_convert_for_all_real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for converting a 'StringView' to 'double'.
 */
class ARCCORE_BASE_EXPORT StringViewToDoubleConverter
{
 public:

  static Int64 _getDoubleValueWithFromChars(double& v, StringView s);
  static Int64 _getDoubleValue(double& v, StringView s);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for converting a 'StringView' to an integral type.
 *
 * The getValue() methods are similar in semantics to the
 * builtInGetValue() functions of Arcane with one exception: they
 * initialize the argument with the value 0 or false even in case of an error.
 */
class ARCCORE_BASE_EXPORT StringViewToIntegral
{
 public:

  static bool getValue(double& v, StringView s);
  static bool getValue(int& v, StringView s);
  static bool getValue(long& v, StringView s);
  static bool getValue(long long& v, StringView s);
  static bool getValue(bool& v, StringView s);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT StringView
_removeLeadingSpaces(StringView s, Int64 pos = 0);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
