// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* String.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Unicode character string.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRING_H
#define ARCCORE_BASE_STRING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#include <string>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//class StringFormatterArg;
//class StringBuilder;
//class StringImpl;
//class StringView;
//class StringUtilsImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Unicode character string.
 *
 * This class allows managing a character string using either UTF-8 or
 * UTF-16 encoding. Note that UTF-16 encoding is obsolete and will be removed
 * in a later version when C++20 is available.
 *
 * All methods using `const char*` as arguments assume that the encoding used
 * is UTF-8.
 *
 * Instances of this class are immutable.
 *
 * This class is similar to std::string but with the following differences:
 * - The \a String class uses UTF-8 encoding, whereas for std::string, the
 *   encoding is undefined.
 * - Unlike std::string, it is currently not possible to preserve null
 *   characters inside a \a String.
 * - For String, there is a distinction between a null string and an empty
 *   string.
 *   The String::String() constructor creates a null string, while
 *   String::String("") creates an empty string. If the string is null,
 *   calls to view() or toStdStringView() return an empty string.
 *
 * When C++20 is available, the \a String class will correspond to the
 * std::optional<std::u8string> type.
 *
 * For performance reasons, when building a string piece by piece, it is
 * preferable to use the 'StringBuilder' class.
 */
class ARCCORE_BASE_EXPORT String
{
 public:

  friend ARCCORE_BASE_EXPORT bool operator<(const String& a, const String& b);
  friend class StringBuilder;
  friend class StringUtilsImpl;

 public:

  //! Creates a null string
  String() {}
  /*!
   * \brief Creates a string from \a str in UTF-8 encoding
   *
   * \warning Attention, the string is assumed to have infinite constant validity
   * (i.e., it is a compile-time constant string.
   * If the string passed as an argument can be deallocated,
   * use String(std::string_view) instead.
   */
  String(const char* str)
  : m_const_ptr(str)
  {
    if (str)
      m_const_ptr_size = std::string_view(str).size();
  }
  //! Creates a string from \a str in UTF-8 encoding
  String(char* str);
  //! Creates a string from \a str in UTF-8 encoding
  ARCCORE_DEPRECATED_2019("Use String::String(StringView) instead")
  String(const char* str, bool do_alloc);
  //! Creates a string from \a str in UTF-8 encoding
  ARCCORE_DEPRECATED_2019("Use String::String(StringView) instead")
  String(const char* str, Integer len);
  //! Creates a string from \a str in UTF-8 encoding
  String(std::string_view str);
  //! Creates a string from \a str in UTF-8 encoding
  String(StringView str);
  //! Creates a string from \a str in UTF-8 encoding
  String(const std::string& str);
  //! Creates a string from \a str in UTF-16 encoding
  String(const UCharConstArrayView& ustr);
  //! Creates a string from \a str in UTF-8 encoding
  String(const Span<const Byte>& ustr);
  //! Creates a string from \a str in UTF-8 encoding
  //String(const Span<Byte>& ustr);
  //! Creates a string from \a str in UTF-8 encoding
  explicit String(StringImpl* impl);
  //! Creates a string from \a str
  String(const String& str);
  //! Creates a string from \a str
  String(String&& str)
  : m_p(str.m_p)
  , m_const_ptr(str.m_const_ptr)
  , m_const_ptr_size(str.m_const_ptr_size)
  {
    str._resetFields();
  }

  //! Copies \a str into this instance.
  String& operator=(const String& str);
  //! Copies \a str into this instance.
  String& operator=(String&& str);
  //! Copies \a str into this instance.
  String& operator=(StringView str);
  /*!
   * \brief References \a str encoded in UTF-8 in this instance.
   *
   * \warning Attention, the string is assumed to have infinite constant validity
   * (i.e., it is a compile-time constant string.
   * If the string passed as an argument can be deallocated,
   * use String::operator=(std::string_view) instead.
   */
  String& operator=(const char* str)
  {
    m_const_ptr = str;
    m_const_ptr_size = 0;
    if (m_const_ptr)
      m_const_ptr_size = std::string_view(str).size();
    _removeReferenceIfNeeded();
    m_p = nullptr;
    return (*this);
  }
  //! Copies \a str encoded in UTF-8 into this instance.
  String& operator=(std::string_view str);
  //! Copies \a str encoded in UTF-8 into this instance.
  String& operator=(const std::string& str);

  //! Frees resources.
  ~String()
  {
    _removeReferenceIfNeeded();
  }

 public:

  /*!
   * \brief Returns a view of the current string.
   *
   * The encoding used is UTF-8.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance. The returned view
   * should not be retained.
   */
  operator StringView() const;

 public:

  static String fromUtf8(Span<const Byte> bytes);

 public:

  /*!
   * \brief Returns the conversion of the instance into UTF-16 encoding.
   *
   * The returned array always contains a terminal zero if the string is not
   * null-terminated. Therefore, the size of any non-null string is
   * the array size minus 1.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance.
   *
   * \deprecated Use StringUtils::asUtf16BE() instead. Note that
   * the StringUtils::asUtf16BE() function does not contain a terminal 0x00.
   */
  [[deprecated("Y2022: Use StringUtils::asUtf16BE() instead")]]
  ConstArrayView<UChar> utf16() const;

  /*!
   * \brief Returns the conversion of the instance into UTF-8 encoding.
   *
   * The returned array always contains a terminal zero if the string is not
   * null-terminated. Therefore, the size of any non-null string is
   * the array size minus 1.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance.
   */
  ByteConstArrayView utf8() const;

  /*!
   * \brief Returns the conversion of the instance into UTF-8 encoding.
   *
   * \a bytes().size() corresponds to the length of the character string but
   * the returned view always contains a terminal '\0'.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance.
   */
  Span<const Byte> bytes() const;

  /*!
   * \brief Returns the conversion of the instance into UTF-8 encoding.
   *
   * If null() is true, returns the empty string. Otherwise, this method is equivalent
   * to calling bytes().data(). There is always a terminal '\0' at the end of the
   * returned string.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance.
   */
  const char* localstr() const;

 public:

  /*!
   * \brief Returns an STL view of the current string.
   *
   * The encoding used is UTF-8.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance. The returned view
   * should not be retained.
   */
  std::string_view toStdStringView() const;

  /*!
   * \brief Returns a view of the current string.
   *
   * The encoding used is UTF-8.
   *
   * \warning The instance remains the owner of the returned value and this value
   * is invalidated by any modification of this instance. The returned view
   * should not be retained.
   */
  StringView view() const;

 public:

  //! Clones this string.
  String clone() const;

  /*!
   * \brief Performs whitespace character normalization.
   *
   * All whitespace characters are replaced by space characters #x20,
   * namely #xD (Carriage Return), #xA (Line Feed), and #x9 (Tabulation).
   * This corresponds to the xs:replace attribute of XMLSchema 1.0
   */
  static String replaceWhiteSpace(const String& rhs);

  /*!
   * \brief Performs whitespace character normalization.
   *
   * The behavior is identical to replaceWhiteSpace() plus:
   * - replacement of all consecutive whitespaces by a single one.
   * - removal of whitespaces at the beginning and end of the string.
   * This corresponds to the xs:collapse attribute of XMLSchema 1.0
   */
  static String collapseWhiteSpace(const String& rhs);

  //! Transforms all characters in the string to uppercase.
  String upper() const;

  //! Transforms all characters in the string to lowercase.
  String lower() const;

  //! Returns \a true if the string is null.
  bool null() const;

  //! Returns the length of the string in 32 bits.
  ARCCORE_DEPRECATED_2019("Use method String::length() instead")
  Integer len() const;

  //! Returns the length of the string.
  Int64 length() const;

  //! True if the string is empty (null or "")
  bool empty() const;

  //! Calculates a hash value for this character string
  Int32 hashCode() const;

  //! Writes the string in UTF-8 format to the stream \a o
  void writeBytes(std::ostream& o) const;

 public:

  /*!
   * \brief Compares two unicode strings.
   * \retval true if they are equal,
   * \retval false otherwise.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const String& a, const String& b);

  /*!
   * \brief Compares two unicode strings.

   * \retval true if they are different,
   * \retval false if they are equal.
   * \relate String
   */
  friend bool operator!=(const String& a, const String& b)
  {
    return !operator==(a, b);
  }

  //! String output operator
  friend ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o, const String&);
  //! String input operator
  friend ARCCORE_BASE_EXPORT std::istream& operator>>(std::istream& o, String&);

  /*!
   * \brief Compares two unicode strings.
   * \retval true if they are equal,
   * \retval false otherwise.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const char* a, const String& b);

  /*!
   * \brief Compares two unicode strings.
   * \retval true if they are different,
   * \retval false if they are equal.
   */
  inline friend bool operator!=(const char* a, const String& b)
  {
    return !operator==(a, b);
  }

  /*!
   * \brief Compares two unicode strings.
   * \retval true if they are equal,
   * \retval false otherwise.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const String& a, const char* b);

  /*!
   * \brief Compares two unicode strings.
   * \retval true if they are different,
   * \retval false if they are equal.
   */
  inline friend bool operator!=(const String& a, const char* b)
  {
    return !operator==(a, b);
  }

  //! Adds two strings.
  friend ARCCORE_BASE_EXPORT String operator+(const char* a, const String& b);

  friend ARCCORE_BASE_EXPORT bool operator<(const String& a, const String& b);

 public:

  //! Returns the concatenation of this string with the string
  //! \a str encoded in UTF-8
  String operator+(const char* str) const
  {
    if (!str)
      return (*this);
    return operator+(std::string_view(str));
  }
  //! Returns the concatenation of this string with the string
  //! \a str encoded in UTF-8
  String operator+(std::string_view str) const;
  //! Returns the concatenation of this string with the string
  //! \a str encoded in UTF-8
  String operator+(const std::string& str) const;
  //! Returns the concatenation of this string with the string
  //! \a str.
  String operator+(const String& str) const;
  String operator+(unsigned long v) const;
  String operator+(unsigned int v) const;
  String operator+(double v) const;
  String operator+(long double v) const;
  String operator+(int v) const;
  String operator+(long v) const;
  String operator+(unsigned long long v) const;
  String operator+(long long v) const;
  String operator+(const APReal& v) const;

  static String fromNumber(unsigned long v);
  static String fromNumber(unsigned int v);
  static String fromNumber(double v);
  static String fromNumber(double v, Integer nb_digit_after_point);
  static String fromNumber(long double v);
  static String fromNumber(int v);
  static String fromNumber(long v);
  static String fromNumber(unsigned long long v);
  static String fromNumber(long long v);
  static String fromNumber(const APReal& v);

 public:

  static String format(const String& str);
  static String format(const String& str, const StringFormatterArg& arg1);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7,
                       const StringFormatterArg& arg8);
  static String format(const String& str, const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4,
                       const StringFormatterArg& arg5,
                       const StringFormatterArg& arg6,
                       const StringFormatterArg& arg7,
                       const StringFormatterArg& arg8,
                       const StringFormatterArg& arg9);
  static String concat(const StringFormatterArg& arg1);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3);
  static String concat(const StringFormatterArg& arg1,
                       const StringFormatterArg& arg2,
                       const StringFormatterArg& arg3,
                       const StringFormatterArg& arg4);

  //! Standard plural form by adding an 's'
  static String plural(const Integer n, const String& str, const bool with_number = true);
  //! Special plural form by variant
  static String plural(const Integer n, const String& str, const String& str2, const bool with_number = true);

  //! Indicates if the string contains \a s
  bool contains(const String& s) const;

  //! Indicates if the string starts with the characters of \a s
  bool startsWith(const String& s) const;

  //! Indicates if the string ends with the characters of \a s
  bool endsWith(const String& s) const;

  //! Substring starting at position \a pos
  String substring(Int64 pos) const;

  //! Substring starting at position \a pos and of length \a len
  String substring(Int64 pos, Int64 len) const;

  static String join(String delim, ConstArrayView<String> strs);

  //! Splits the string based on the character \a c
  template <typename StringContainer> void
  split(StringContainer& str_array, char c) const
  {
    const String& str = *this;
    //TODO: pass through String::bytes().
    const char* str_str = str.localstr();
    Int64 offset = 0;
    Int64 len = str.length();
    for (Int64 i = 0; i < len; ++i) {
      // GG: temporarily reverts to the old semantics (equivalent to strtok())
      // and removes the IFPEN modification because it causes too many incompatibilities with
      // the existing code. Note that the implementation of the old semantics
      // has several bugs:
      //   1. in the case where the delimiter is repeated 3 times or more consecutively.
      // For example, 'X:::Y' returns {'X',':','Y'} instead of
      // {'X','Y'}
      //   2. If it starts with the delimiter, the delimiter is returned:
      // With ':X:Y', we return {':X','Y'} instead of {'X','Y'}
      //if (str_str[i]==c){
      if (str_str[i] == c && i != offset) {
        str_array.push_back(std::string_view(str_str + offset, i - offset));
        offset = i + 1;
      }
    }
    if (len != offset)
      str_array.push_back(std::string_view(str_str + offset, len - offset));
  }

 public:

  /*!
   * \brief Displays the internal information of the class.
   *
   * This method is only useful for debugging Arccore
   */
  void internalDump(std::ostream& ostr) const;

 private:

  mutable StringImpl* m_p = nullptr; //!< Class implementation
  mutable const char* m_const_ptr = nullptr;
  mutable Int64 m_const_ptr_size = 0; //!< String length if constant (-1 otherwise)

  void _checkClone() const;
  bool isLess(const String& s) const;
  String& _append(const String& str);
  // Only call if 'm_const_ptr' is not null otherwise m_const_ptr_size is (-1)
  std::string_view _viewFromConstChar() const
  {
    return std::string_view(m_const_ptr, m_const_ptr_size);
  }
  void _removeReference();
  ConstArrayView<UChar> _internalUtf16BE() const;
  void _resetFields()
  {
    m_p = nullptr;
    m_const_ptr = nullptr;
    m_const_ptr_size = 0;
  }
  void _copyFields(const String& str)
  {
    m_p = str.m_p;
    m_const_ptr = str.m_const_ptr;
    m_const_ptr_size = str.m_const_ptr_size;
  }

  /*!
   * \brief Removes the reference to the implementation if it is not null.
   */
  void _removeReferenceIfNeeded()
  {
    if (m_p)
      _removeImplReference();
  }

  /*!
   * \brief Removes the reference to the implementation.
   * \pre m_p != nullptr
   */
  void _removeImplReference();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename U>
class StringFormatterArgToString
{
 public:

  static void toString(const U& v, String& s)
  {
    std::ostringstream ostr;
    ostr << v;
    s = ostr.str();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class used to format a character string.
 */
class ARCCORE_BASE_EXPORT StringFormatterArg
{
 public:

  template <typename U>
  StringFormatterArg(const U& avalue)
  {
    StringFormatterArgToString<U>::toString(avalue, m_str_value);
  }
  StringFormatterArg(Real avalue)
  {
    _formatReal(avalue);
  }
  StringFormatterArg(const String& s)
  : m_str_value(s)
  {
  }

 public:

  const String& value() const { return m_str_value; }

 private:

  String m_str_value;

 private:

  void _formatReal(Real avalue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Span<const std::byte>
asBytes(const String& v)
{
  return asBytes(v.bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
