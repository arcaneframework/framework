// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueConvert.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueConvert.h"

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/internal/ValueConvertInternal.h"

#include "arccore/base/internal/ConvertInternal.h"

#include <charconv>

// TODO: Pour builtInGetValue(), retourner `true` si la chaîne en entrée est vide.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
  /*!
   * \brief Retourne \a s converti en \a 'const char*'.
   *
   * \warning Si la valeur retournée est utilisée pour une fonction C,
   * il faut être sur que \a s a un '\0' terminal.
   */
  const char* _stringViewData(StringView s)
  {
    return reinterpret_cast<const char*>(s.bytes().data());
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::StringViewInputStream::
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(double& v, StringView s)
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

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(BFloat16& v, StringView s)
{
  float z = 0.0;
  bool r = builtInGetValue(z, s);
  v = z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Float16& v, StringView s)
{
  float z = 0.0;
  bool r = builtInGetValue(z, s);
  v = z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Float128& v, StringView s)
{
  // Pour l'instant (12/2024), il n'y a pas de fonctions natives pour lire un Float128.
  // On utilise donc un 'long double'.
  // TODO: à implémenter correctement
  long double z = 0.0;
  bool r = builtInGetValue(z, s);
  v = z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(float& v, StringView s)
{
  double z = 0.;
  bool r = builtInGetValue(z, s);
  v = (float)z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtol(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(int& v, StringView s)
{
  long z = 0;
  bool r = builtInGetValue(z, s);
  v = (int)z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(short& v, StringView s)
{
  long z = 0;
  bool r = builtInGetValue(z, s);
  v = (short)z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(unsigned long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtoul(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(unsigned int& v, StringView s)
{
  unsigned long z = 0;
  bool r = builtInGetValue(z, s);
  v = (unsigned int)z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(unsigned short& v, StringView s)
{
  unsigned long z = 0;
  bool r = builtInGetValue(z, s);
  v = (unsigned short)z;
  return r;
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(long long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtoll(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2& v, StringView s)
{
  if (Convert::Impl::ConvertPolicy::isUseSameConvertForAllReal()) {
    s = Convert::Impl::_removeLeadingSpaces(s);
    v = {};
    const bool is_verbose = Convert::Impl::ConvertPolicy::verbosity() > 0;
    if (is_verbose)
      std::cout << "Try Read Real2: '" << s << "'\n";
    Int64 p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValue(v.x, s);
    if (p == (-1))
      return true;
    s = Convert::Impl::_removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VX=" << v.x << " remaining_s='" << s << "'\n";
    p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValue(v.y, s);
    return (p == (-1) || (p != s.size()));
  }
  return impl::builtInGetValueGeneric(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3& v, StringView s)
{
  if (Convert::Impl::ConvertPolicy::isUseSameConvertForAllReal()){
    s = Convert::Impl::_removeLeadingSpaces(s);
    v = {};
    const bool is_verbose = Convert::Impl::ConvertPolicy::verbosity() > 0;
    if (is_verbose)
      std::cout << "Try Read Real3: '" << s << "'\n";
    Int64 p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValue(v.x, s);
    if (p == (-1) || (p == s.size()))
      return true;
    s = Convert::Impl::_removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VX=" << v.x << " remaining_s='" << s << "'\n";
    p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValue(v.y, s);
    if (p == (-1) || (p == s.size()))
      return true;
    s = Convert::Impl::_removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VY=" << v.x << " remaining_s='" << s << "'\n";
    p = Convert::Impl::StringViewToDoubleConverter::_getDoubleValue(v.z, s);
    return (p == (-1) || (p != s.size()));
  }
  return impl::builtInGetValueGeneric(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2x2& v, StringView s)
{
  return impl::builtInGetValueGeneric(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3x3& v, StringView s)
{
  return impl::builtInGetValueGeneric(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int128& v, StringView s)
{
  // Pour l'instant (12/2024), il n'y a pas de fonctions natives pour lire un Int128.
  // On utilise donc un 'Int64' en attendant.
  // TODO: il existe des exemples sur internet. A implémenter correctement
  long long v2 = 0;
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v2 = ::strtoll(ptr, &ptr2, 0);
  v = v2;
  return (ptr2 != (ptr + s.length()));
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(unsigned long long& v, StringView s)
{
  const char* ptr = _stringViewData(s);
  char* ptr2 = 0;
  v = ::strtoull(ptr, &ptr2, 0);
  return (ptr2 != (ptr + s.length()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_REAL_NOT_BUILTIN
template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real& v, StringView s)
{
#if 0
  double vz = 0.0;
  if (builtInGetValue(vz,s))
    return true;
  v = vz;
  cout << "** CONVERT DOUBLE TO REAL s=" << s << " vz=" << vz << " v=" << v << '\n';
  return false;
#endif
  double vz = 0.0;
  if (builtInGetValue(vz, s))
    return true;
  v = Real((char*)s.localstr(), 1000);
  cout << "** CONVERT DOUBLE TO REAL s=" << s << '\n';
  return false;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> bool
builtInGetValue(String& v, StringView s)
{
  v = s;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  bool _builtInGetBoolArrayValue(BoolArray& v, StringView s)
  {
    // Le type 'bool' est un peu spécial car il doit pouvoir lire les
    // valeurs comme 'true' ou 'false'.
    // On le lit donc comme un 'StringUniqueArray', puis on converti en bool
    //cout << "** GET BOOL ARRAY V=" << s << '\n';
    //return builtInGetArrayValue(v,s);

    StringUniqueArray sa;
    if (builtInGetValue(sa, s))
      return true;
    for (Integer i = 0, is = sa.size(); i < is; ++i) {
      bool read_val = false;
      if (builtInGetValue(read_val, sa[i]))
        return true;
      v.add(read_val);
    }
    return false;
  }

  bool
  _builtInGetStringArrayValue(StringArray& v, StringView s)
  {
    std::string s2;
    String read_val = String();
    impl::StringViewInputStream svis(s);
    std::istream& sbuf = svis.stream();
    while (!sbuf.eof()) {
      sbuf >> s2;
      //cout << " ** CHECK READ v='" << s2 << "' '" << sv << "'\n";
      if (sbuf.bad()) // non-recoverable error
        return true;
      if (sbuf.fail()) // recoverable error : this means good conversion
        return false;
      read_val = StringView(s2.c_str());
      v.add(read_val);
    }
    return false;
  }
} // namespace

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(RealArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2x2Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3x3Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int8Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int16Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int32Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int64Array& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(BoolArray& v, StringView s)
{
  return _builtInGetBoolArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(StringArray& v, StringView s)
{
  return _builtInGetStringArrayValue(v, s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(RealUniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2x2UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3x3UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int8UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int16UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int32UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int64UniqueArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(BoolUniqueArray& v, StringView s)
{
  return _builtInGetBoolArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(StringUniqueArray& v, StringView s)
{
  return _builtInGetStringArrayValue(v, s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(RealSharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real2x2SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3x3SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int8SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int16SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int32SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Int64SharedArray& v, StringView s)
{
  return builtInGetArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(BoolSharedArray& v, StringView s)
{
  return _builtInGetBoolArrayValue(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(StringSharedArray& v, StringView s)
{
  return _builtInGetStringArrayValue(v, s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  template <class T> inline bool
  _builtInPutValue(const T& v, String& s)
  {
    OStringStream ostr;
    ostr() << v;
    if (ostr().fail() || ostr().bad())
      return true;
    s = ostr.str();
    return false;
  }
  template <class T> inline bool
  _builtInPutArrayValue(Span<const T> v, String& s)
  {
    OStringStream ostr;
    for (Int64 i = 0, n = v.size(); i < n; ++i) {
      if (i != 0)
        ostr() << ' ';
      ostr() << v[i];
    }
    if (ostr().fail() || ostr().bad())
      return true;
    s = ostr.str();
    return false;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool builtInPutValue(const String& v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(double v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(float v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(int v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(unsigned int v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(long v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(long long v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(short v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(unsigned short v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(unsigned long v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(unsigned long long v, String& s)
{
  return _builtInPutValue(v, s);
}
#ifdef ARCANE_REAL_NOT_BUILTIN
bool builtInPutValue(Real v, String& s)
{
  return _builtInPutValue(v, s);
}
#endif
bool builtInPutValue(Real2 v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(Real3 v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(const Real2x2& v, String& s)
{
  return _builtInPutValue(v, s);
}
bool builtInPutValue(const Real3x3& v, String& s)
{
  return _builtInPutValue(v, s);
}

bool builtInPutValue(Span<const Real> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Real2> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Real3> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Real2x2> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Real3x3> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Int16> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Int32> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const Int64> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const bool> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
bool builtInPutValue(Span<const String> v, String& s)
{
  return _builtInPutArrayValue(v, s);
}
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
