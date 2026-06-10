// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.cc                                                  (C) 2000-2026 */
/*                                                                           */
/* Functions to convert a character string into a given type.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/internal/ConvertInternal.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Convert.h"
#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"

#include <charconv>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{
namespace
{
  /*!
   * \brief Returns `s` converted to `const char*`.
   *
   * \warning If the returned value is used for a C function,
   * you must ensure that `s` has a terminating '\0'.
   */
  const char* _stringViewData(StringView s)
  {
    return reinterpret_cast<const char*>(s.bytes().data());
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns a view after removing leading whitespace characters.
 *
 * A whitespace character is a character for which std::isspace() is true.
 * `pos` indicates the position in `s` from which
 * the whitespace characters are searched.
 */
StringView _removeLeadingSpaces(StringView s, Int64 pos)
{
  Span<const Byte> bytes = s.bytes();
  Int64 nb_byte = bytes.size();
  // Remove potential spaces
  for (; pos < nb_byte; ++pos) {
    int charv = static_cast<unsigned char>(bytes[pos]);
    if (std::isspace(charv) == 0)
      break;
  }
  return s.subView(pos, nb_byte);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringViewInputStream::
StringViewInputStream(StringView v)
: m_view(v)
, m_stream(this)
{
  auto b = v.bytes();
  char* begin_ptr = const_cast<char*>(reinterpret_cast<const char*>(b.data()));
  char* end_ptr = begin_ptr + b.size();
  setg(begin_ptr, begin_ptr, end_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ConvertPolicy::m_verbosity = 0;
bool ConvertPolicy::m_use_from_chars = true;
bool ConvertPolicy::m_use_same_convert_for_all_real = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts `s` to a double.
 *
 * Uses std::from_chars() if `global_use_from_chars` is true.
 * Otherwise, uses strtod().
 */
Int64 StringViewToDoubleConverter::
_getDoubleValue(double& v, StringView s)
{
  if (ConvertPolicy::isUseFromChars()) {
    s = _removeLeadingSpaces(s);
    Int64 p = _getDoubleValueWithFromChars(v, s);
    return p;
  }

  const char* ptr = _stringViewData(s);
#ifdef WIN32
  if (s == "infinity" || s == "inf") {
    v = std::numeric_limits<double>::infinity();
    return s.size();
  }
#endif
  char* ptr2 = nullptr;
  if (ptr)
    v = ::strtod(ptr, &ptr2);
  return (ptr2 - ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a character string to a double.
 *
 * Converts `s` to a double and stores the value in `v`.
 * There must be no whitespace characters at the beginning of `s`.
 *
 * The behavior of this method is identical to std::strtod()
 * with the 'C' locale if running in C++20. Otherwise, it is identical
 * to std::strtod() with the current locale (which can change, for example
 * the decimal separator). The reference documentation is
 * here: https://en.cppreference.com/w/cpp/utility/from_chars.
 *
 * \retval (-1) if the conversion failed.
 * \retval the position in `s` of the last character read plus 1.
 */
Int64 StringViewToDoubleConverter::
_getDoubleValueWithFromChars(double& v, StringView s)
{
  // NOTE: if we want the same behavior as 'strtod',
  // we assume the caller has removed leading whitespace from s.
  auto bytes = s.bytes();
  Int64 size = bytes.size();
  if (size == 0)
    // NOTE: With the historical version of Arcane (before 3.15) there
    // was no error returned when converting an empty string.
    // Apparently this was never used, so there is no problem correcting this bug.
    return (-1);
  const char* orig_data = reinterpret_cast<const char*>(bytes.data());
  const char* last_ptr = nullptr;
  std::chars_format fmt = std::chars_format::general;
  const char* data = orig_data;
  bool do_negatif = false;
  const bool is_verbose = ConvertPolicy::verbosity() > 0;

  // std::from_chars() does not support '+' at the beginning while
  // 'strto*' does.
  if (bytes[0] == '+') {
    ++data;
    --size;
    bytes = bytes.subspan(1, size);
  }
  // std::from_chars() can read values in hexadecimal format
  // but it must not contain '0x' or '0X' at the beginning, unlike
  // std::strtod(). We detect this case and start the conversion
  // after '0x' or '0X'.
  // Detects '-0x' or '-0X'
  if (size >= 3 && (bytes[0] == '-') && (bytes[1] == '0') && (bytes[2] == 'x' || bytes[2] == 'X')) {
    fmt = std::chars_format::hex;
    data += 3;
    do_negatif = true;
  }
  // Detects '0x' or '0X'
  else if (size >= 2 && (bytes[0] == '0') && (bytes[1] == 'x' || bytes[1] == 'X')) {
    fmt = std::chars_format::hex;
    data += 2;
  }
  // General case
  {
    auto [ptr, ec] = std::from_chars(data, data + size, v, fmt);
    last_ptr = ptr;
    if (is_verbose)
      std::cout << "FromChars:TRY GET_DOUBLE data=" << data << " v=" << v << " is_ok=" << (ec == std::errc()) << "\n";
    if (ec != std::errc())
      return (-1);
  }
  // Account for the '-' sign if requested
  if (do_negatif)
    v = -v;
  if (is_verbose) {
    char* ptr2 = nullptr;
    double v2 = ::strtod(orig_data, &ptr2);
    std::cout << "FromChars: COMPARE GET_DOUBLE via strtod v2=" << v2 << " pos=" << (ptr2 - orig_data) << "\n";
  }
  return (last_ptr - orig_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringViewToIntegral::
getValue(double& v, StringView s)
{
  if (Convert::Impl::ConvertPolicy::isUseFromChars()) {
    s = Convert::Impl::_removeLeadingSpaces(s);
    Int64 p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValueWithFromChars(v, s);
    return (p == (-1) || (p != s.size()));
  }

  const char* ptr = _stringViewData(s);
#ifdef WIN32
  if (s == "infinity" || s == "inf") {
    v = std::numeric_limits<double>::infinity();
    return false;
  }
#endif
  char* ptr2 = nullptr;
  v = ::strtod(ptr, &ptr2);
  return (ptr2 != (ptr + s.length()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringViewToIntegral::
getValue(long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtol(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringViewToIntegral::
getValue(int& v, StringView s)
{
  long z = 0;
  bool is_bad = getValue(z, s);
  if (!is_bad)
    // TODO: Perform validity check before cast
    v = static_cast<int>(z);
  return is_bad;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringViewToIntegral::
getValue(long long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtoll(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringViewToIntegral::
getValue(bool& v, StringView s)
{
  v = false;
  int x = 0;
  if (getValue(x, s))
    return true;
  v = (x != 0);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const char* _typeName(Int32)
  {
    return "Int32";
  }
  const char* _typeName(Int64)
  {
    return "Int64";
  }
  const char* _typeName(Real)
  {
    return "Real";
  }
} // namespace

template <typename T> std::optional<T>
ScalarType<T>::tryParse(StringView s)
{
  T v;
  if (s.empty())
    return std::nullopt;
  bool is_bad = Impl::StringViewToIntegral::getValue(v, s);
  if (is_bad)
    return std::nullopt;
  return v;
}

template <typename T> std::optional<T>
ScalarType<T>::tryParseFromEnvironment(StringView s, bool throw_if_invalid)
{
  String env_value = Platform::getEnvironmentVariable(s);
  if (env_value.null())
    return std::nullopt;
  auto v = tryParse(env_value);
  if (!v && throw_if_invalid)
    ARCCORE_FATAL("Invalid value '{0}' for environment variable {1}. Can not convert to type '{2}'",
                  env_value, s, _typeName(T{}));
  return v;
}

template class ScalarType<Int32>;
template class ScalarType<Int64>;
template class ScalarType<Real>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
