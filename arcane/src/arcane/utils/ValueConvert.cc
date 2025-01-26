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
#include "arcane/utils/internal/ValueConvertInternal.h"

// En théorie std::from_chars() est disponible avec le C++17 mais pour
// GCC cela n'est implémenté pour les flottants qu'à partir de GCC 11.
// Comme c'est la version requise pour le C++20, on n'active cette fonctionnalité
// qu'à partir du C++20.
#if defined(ARCANE_HAS_CXX20)
#define ARCANE_USE_FROMCHARS
#endif

#if defined(ARCANE_USE_FROMCHARS)
#include <charconv>
#endif

// TODO: Pour builtInGetValue(), retourner `true` si la chaîne en entrée est vide.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
  const char* _stringViewData(StringView s)
  {
    return reinterpret_cast<const char*>(s.bytes().data());
  }

  /*!
   * \brief Retourne une vue en supprimant les caratères blancs du début.
   *
   * Un caractère blanc est un caractère pour lequel std::isspace() est vrai.
   * \a pos indique la position dans \a s à partir de laquelle
   * on cherche les blancs.
   */
  StringView _removeLeadingSpaces(StringView s, Int64 pos)
  {
    Span<const Byte> bytes = s.bytes();
    Int64 nb_byte = bytes.size();
    // Supprime les espaces potentiels
    for (; pos < nb_byte; ++pos) {
      int charv = static_cast<unsigned char>(bytes[pos]);
      // Visual Studio 2017 or less
#if defined(_MSC_VER) && _MSC_VER <= 1916
      if (std::isspace(charv, std::locale()) != 0)
        break;
#else
      if (!std::isspace(charv) != 0)
        break;
#endif
    }
    return StringView(bytes.subSpan(pos, nb_byte));
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

namespace
{
  Int32 global_value_convert_verbosity = 0;
  bool global_use_from_chars = true;
  bool global_use_same_value_convert_for_all_real = false;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::
arcaneSetIsValueConvertUseFromChars(bool v)
{
  global_use_from_chars = v;
}

void impl::
arcaneSetValueConvertVerbosity(Int32 v)
{
  global_value_convert_verbosity = v;
}

void impl::
arcaneSetUseSameValueConvertForAllReal(bool v)
{
  global_use_same_value_convert_for_all_real = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour convertir une 'StringView' en 'double'.
 */
class StringViewToDoubleConverter
{
 public:

  static Int64 _getDoubleValueWithFromChars(double& v, StringView s);
  static Int64 _getDoubleValue(double& v, StringView s);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti \a s en un double.
 *
 * Utilise std::from_chars() si \a global_use_from_chars est vrai.
 * Sinon, utilise strtod().
 */
Int64 StringViewToDoubleConverter::
_getDoubleValue(double& v, StringView s)
{
#if defined(ARCANE_USE_FROMCHARS)
  if (global_use_from_chars) {
    Int64 p = _getDoubleValueWithFromChars(v, s);
    return p;
  }
#endif

  const char* ptr = _stringViewData(s);
#ifdef WIN32
  if (s == "infinity" || s == "inf") {
    v = std::numeric_limits<double>::infinity();
    return false;
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
 * \brief Converti une chaîne de caractères en un double.
 *
 * Converti \a s en un double et range la valeur dans \a v.
 * Il ne doit pas y avoir de caractères blancs au début de \a s.
 *
 * Le comportement de cette méthode est identique à std::strtod()
 * avec le locale 'C' si on est en C++20. Sinon il est identique
 * à std::strtod() avec le locale actuel (ce qui peut changer par exemple
 * le séparateur décimal). La documentation de référence est
 * ici: https://en.cppreference.com/w/cpp/utility/from_chars.
 *
 * \retval (-1) si la conversion a échouée.
 * \retval la position dans \s du dernier caratère lu plus 1.
 */
Int64 StringViewToDoubleConverter::
_getDoubleValueWithFromChars(double& v, StringView s)
{
#if defined(ARCANE_USE_FROMCHARS)
  // ATTENTION: il ne faut pas d'espace en début de \a s
  auto bytes = s.bytes();
  Int64 size = bytes.size();
  if (size == 0)
    // NOTE: Avec la version historique d'Arcane (avant la 3.15) il
    // n'y avait pas d'erreur retournée lorsqu'on converti une chaîne vide.
    // A priori cela n'était jamais utilisé donc cela ne pose pas de
    // problème de corriger ce bug.
    return (-1);
  const char* orig_data = reinterpret_cast<const char*>(bytes.data());
  const char* last_ptr = nullptr;
  std::chars_format fmt = std::chars_format::general;
  const char* data = orig_data;
  bool do_negatif = false;
  const bool is_verbose = global_value_convert_verbosity > 0;
  // std::from_chars() peut lire les valeurs au format hexadécimal
  // mais il ne doit pas contenir le '0x' ou '0X' du début, contrairement
  // à std::strtod(). On détecte ce cas et on commence la conversion
  // après le '0x' ou '0X'.

  // Détecte '-0x' ou '-0X'
  if (size >= 3 && (bytes[0] == '-') && (bytes[1] == '0') && (bytes[2] == 'x' || bytes[2] == 'X')) {
    fmt = std::chars_format::hex;
    data += 3;
    do_negatif = true;
  }
  // Détecte '0x' ou '0X'
  else if (size >= 2 && (bytes[0] == '0') && (bytes[1] == 'x' || bytes[1] == 'X')) {
    fmt = std::chars_format::hex;
    data += 2;
  }
  // Cas général
  {
    auto [ptr, ec] = std::from_chars(data, data + size, v, fmt);
    last_ptr = ptr;
    if (is_verbose)
      std::cout << "FromChars:TRY GET_DOUBLE data=" << data << " v=" << v << " is_ok=" << (ec == std::errc()) << "\n";
    if (ec != std::errc())
      return (-1);
  }
  // Prend en compte le signe '-' si demandé
  if (do_negatif)
    v = -v;
  if (is_verbose) {
    char* ptr2 = nullptr;
    double v2 = ::strtod(orig_data, &ptr2);
    std::cout << "FromChars: COMPARE GET_DOUBLE via strtod v2=" << v2 << " pos=" << (ptr2 - orig_data) << "\n";
  }
  return (last_ptr - orig_data);
#else
  ARCANE_THROW(NotSupportedException, "using std::from_chars() is not available on this platform");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(double& v, StringView s)
{
#if defined(ARCANE_USE_FROMCHARS)
  if (global_use_from_chars) {
    Int64 p = StringViewToDoubleConverter::_getDoubleValueWithFromChars(v, s);
    return (p == (-1) || (p != s.size()));
  }
#endif

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
  if (global_use_same_value_convert_for_all_real) {
    // ATTENTION: Pour l'instant ce nouveau mécanisme ne tolère pas
    // les espaces en début de \a s.
    v = {};
    const bool is_verbose = global_value_convert_verbosity > 0;
    if (is_verbose)
      std::cout << "Try Read Real2: '" << s << "'\n";
    Int64 p = StringViewToDoubleConverter::_getDoubleValue(v.x, s);
    if (p == (-1))
      return true;
    s = _removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VX=" << v.x << " remaining_s='" << s << "'\n";
    p = StringViewToDoubleConverter::_getDoubleValue(v.y, s);
    return (p == (-1) || (p != s.size()));
  }
  return impl::builtInGetValueGeneric(v, s);
}

template <> ARCANE_UTILS_EXPORT bool
builtInGetValue(Real3& v, StringView s)
{
  if (global_use_same_value_convert_for_all_real) {
    // ATTENTION: Pour l'instant ce nouveau mécanisme ne tolère pas
    // les espaces en début de \a s.
    v = {};
    const bool is_verbose = global_value_convert_verbosity > 0;
    if (is_verbose)
      std::cout << "Try Read Real3: '" << s << "'\n";
    Int64 p = StringViewToDoubleConverter::_getDoubleValue(v.x, s);
    if (p == (-1) || (p == s.size()))
      return true;
    s = _removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VX=" << v.x << " remaining_s='" << s << "'\n";
    p = StringViewToDoubleConverter::_getDoubleValue(v.y, s);
    if (p == (-1) || (p == s.size()))
      return true;
    s = _removeLeadingSpaces(s, p);
    if (is_verbose)
      std::cout << "VY=" << v.x << " remaining_s='" << s << "'\n";
    p = StringViewToDoubleConverter::_getDoubleValue(v.z, s);
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
