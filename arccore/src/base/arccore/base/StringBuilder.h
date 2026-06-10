// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringBuilder.h                                             (C) 2000-2025 */
/*                                                                           */
/* Unicode character string constructor.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGBUILDER_H
#define ARCCORE_BASE_STRINGBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <string>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//class String;
//class StringImpl;
//class StringFormatterArg;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Unicode character string constructor.
 *
 * Allows efficient construction of a character string
 * by concatenation.
 *
 * \not_thread_safe
 */
class ARCCORE_BASE_EXPORT StringBuilder
{
 public:

  //! Creates a null string
  StringBuilder()
  : m_p(nullptr)
  , m_const_ptr(nullptr)
  {}
  //! Creates a string from \a str in local encoding
  StringBuilder(const char* str);
  //! Creates a string from \a str in local encoding
  StringBuilder(const char* str, Integer len);
  //! Creates a string from \a str in local encoding
  StringBuilder(const std::string& str);
  //! Creates a string from \a str in Utf16 encoding
  StringBuilder(const UCharConstArrayView& ustr);
  //! Creates a string from \a str in Utf8 encoding
  StringBuilder(const ByteConstArrayView& ustr);
  //! Creates a string from \a str_builder
  StringBuilder(const StringBuilder& str_builder);
  //! Creates a string from \a str in local encoding
  explicit StringBuilder(StringImpl* impl);
  //! Creates a string from \a str
  StringBuilder(const String& str);

  //! Copies \a str into this instance.
  const StringBuilder& operator=(const String& str);
  //! Copies \a str into this instance.
  const StringBuilder& operator=(const char* str);
  //! Copies \a str into this instance.
  const StringBuilder& operator=(const StringBuilder& str);

  ~StringBuilder(); //!< Releases resources.

 public:

  /*!
   * \brief Returns the constructed character string.
   */
  operator String() const;

  /*!
   * \brief Returns the constructed character string.
   */
  String toString() const;

 public:

  //! Appends \a str.
  StringBuilder& append(const String& str);

  //! Clones this string.
  StringBuilder clone() const;

  /*! \brief Performs whitespace character normalization.
   * All whitespace characters are replaced by space characters #x20,
   * namely #xD (Carriage Return), #xA (Line Feed), and #x9 (Tabulation).
   * This corresponds to the xs:replace attribute of XMLSchema 1.0
   */
  StringBuilder& replaceWhiteSpace();

  /*! \brief Performs whitespace character normalization.
   * The behavior is identical to replaceWhiteSpace() plus:
   * - replacement of all consecutive whitespaces with a single one.
   * - removal of whitespaces at the beginning and end of the string.
   * This corresponds to the xs:collapse attribute of XMLSchema 1.0
   */
  StringBuilder& collapseWhiteSpace();

  //! Converts all characters in the string to uppercase.
  StringBuilder& toUpper();

  //! Converts all characters in the string to lowercase.
  StringBuilder& toLower();

  void operator+=(const char* str);
  void operator+=(const String& str);
  void operator+=(unsigned long v);
  void operator+=(unsigned int v);
  void operator+=(double v);
  void operator+=(long double v);
  void operator+=(int v);
  void operator+=(char v);
  void operator+=(long v);
  void operator+=(unsigned long long v);
  void operator+=(long long v);
  void operator+=(const APReal& v);

 public:

  friend ARCCORE_BASE_EXPORT bool operator==(const StringBuilder& a, const StringBuilder& b);
  friend bool operator!=(const StringBuilder& a, const StringBuilder& b)
  {
    return !operator==(a, b);
  }

 public:

  /*!
   * \brief Displays internal class information.
   *
   * This method is only useful for debugging %Arccore
   */
  void internalDump(std::ostream& ostr) const;

 private:

  mutable StringImpl* m_p = nullptr; //!< Class implementation
  mutable const char* m_const_ptr = nullptr;

  void _checkClone() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Output operator for a StringBuilder
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o, const StringBuilder&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
